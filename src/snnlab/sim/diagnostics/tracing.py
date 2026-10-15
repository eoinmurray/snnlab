"""Opt-in, bounded native-training measurements using PyTorch Profiler only.

Monitoring adds detached activity reductions and boundary measurements. Profiling
adds CPU/CUDA operator events. Neither is an uninstrumented performance result.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F

try:
    import resource
except ImportError:  # Windows has no resource module.
    resource = None

_CURRENT = ContextVar("snnlab_trace", default=None)


def current_trace():
    """Internal request-local session; no session exists for disabled tracing."""
    return _CURRENT.get()


@dataclass(frozen=True)
class TraceSpec:
    """Measure a contiguous window of training updates in the current call.

    skip_updates is zero based and excludes earlier updates from capture.
    active_updates is a maximum: short runs report a truncated/empty window.
    """

    output: str | Path
    mode: Literal["off", "monitor", "profile", "both"] = "monitor"
    skip_updates: int = 1
    active_updates: int = 3
    block_size: int = 16
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = False
    detailed_labels: bool = True

    def __post_init__(self):
        if self.mode not in {"off", "monitor", "profile", "both"}:
            raise ValueError("trace mode must be off, monitor, profile or both")
        if not isinstance(self.output, (str, Path)) or not str(self.output):
            raise ValueError("trace output must be a directory path")
        for name, minimum in (
            ("skip_updates", 0),
            ("active_updates", 1),
            ("block_size", 1),
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < minimum:
                raise ValueError(f"trace {name} must be an integer >= {minimum}")
        for name in (
            "record_shapes",
            "profile_memory",
            "with_stack",
            "detailed_labels",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"trace {name} must be boolean")


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _csv(path, rows):
    if not rows:
        path.write_text("")
        return
    names = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _union_us(intervals):
    total, end = 0.0, float("-inf")
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def timeline_metrics(events, wall_s):
    """Derive interval coverage and copy rates from exported Kineto events."""
    kernels, copy_intervals, copies = [], [], []
    kernel_count = 0
    for event in events:
        if event.get("ph") != "X":
            continue
        category = str(event.get("cat", ""))
        start, duration = event.get("ts", 0), event.get("dur", 0)
        if category == "kernel":
            kernel_count += 1
            kernels.append((start, start + duration))
        elif category == "gpu_memcpy":
            copy_intervals.append((start, start + duration))
            args = event.get("args", {})
            size = args.get("bytes", args.get("Bytes"))
            copies.append(
                {
                    "name": event.get("name"),
                    "duration_us": duration,
                    "bytes": size if isinstance(size, (int, float)) else None,
                    "GB_per_s": size / (duration * 1000)
                    if isinstance(size, (int, float)) and duration > 0
                    else None,
                }
            )
    busy = _union_us(kernels)
    all_busy = _union_us(kernels + copy_intervals)
    merged_end, gaps = None, []
    for start, end in sorted(kernels):
        if merged_end is not None and start > merged_end:
            gaps.append(start - merged_end)
        merged_end = end if merged_end is None else max(merged_end, end)
    return {
        "kernel_count": kernel_count,
        "kernel_interval_union_s": busy / 1e6,
        "max_internal_kernel_gap_s": max(gaps) / 1e6 if gaps else None,
        "kernel_and_copy_interval_union_s": all_busy / 1e6,
        "kernel_and_copy_coverage_fraction": all_busy / (wall_s * 1e6)
        if wall_s and (kernels or copy_intervals)
        else None,
        "kernel_coverage_fraction": busy / (wall_s * 1e6)
        if wall_s and kernels
        else None,
        "kernel_coverage_status": "derived" if kernels else "unavailable",
        "kernel_coverage_scope": "union of captured GPU kernel intervals / instrumented capture wall time; not SM utilization or occupancy",
        "copies": copies,
        "copy_bandwidth_scope": "Kineto-reported bytes / copy duration where exported; not device DRAM bandwidth",
    }


class TraceSession:
    """Internal native-trainer integration; prefer ExecutionSpec(tracing=...)."""

    def __init__(self, spec, device):
        self.spec = spec
        self.output = Path(spec.output)
        self.device = torch.device(device)
        self.monitor = spec.mode in {"monitor", "both"}
        self.profile = spec.mode in {"profile", "both"}
        self.active = False
        self.captured = []
        self.phases = []
        self.weights = []
        self.gradients = []
        self.activity = {}
        self.memory = []
        self.profiler = None
        self.profiler_started = False
        self.window_wall_s = 0.0
        self.cpu_s = 0.0
        self.capture_started = None
        self.events = []
        self.dt_ms = 1.0
        self.closed = False

    def __enter__(self):
        self.output.mkdir(parents=True, exist_ok=False)
        self.token = _CURRENT.set(self)
        return self

    def __exit__(self, kind, value, traceback):
        try:
            self.finish(error=str(value) if value is not None else None)
        finally:
            if hasattr(self, "model"):
                self.model._neuron_steps = self.original_steps
            _CURRENT.reset(self.token)

    def bind(self, model):
        self.model = model
        self.original_steps = dict(model._neuron_steps)
        if self.profile and self.spec.detailed_labels:

            def wrap(name, function):
                def step(*args, **kwargs):
                    with self.detail(f"neuron/{name}"):
                        return function(*args, **kwargs)

                return step

            model._neuron_steps = {
                name: wrap(name, function)
                for name, function in self.original_steps.items()
            }
        self.dt_ms = model.plan.dt_ms
        self.spiking = {p["id"] for p in model.plan.populations if p["spiking"]}

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    def begin_update(self, index, epoch, batch, parameters):
        self.active = (
            self.spec.skip_updates
            <= index
            < self.spec.skip_updates + self.spec.active_updates
        )
        if not self.active:
            return
        if self.capture_started is None:
            self._sync()
            if self.device.type == "cuda" and self.monitor:
                torch.cuda.reset_peak_memory_stats(self.device)
            if self.profile:
                activities = [torch.profiler.ProfilerActivity.CPU]
                if self.device.type == "cuda":
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                self.profiler = torch.profiler.profile(
                    activities=activities,
                    record_shapes=self.spec.record_shapes,
                    profile_memory=self.spec.profile_memory,
                    with_stack=self.spec.with_stack,
                    with_flops=True,
                )
                self.profiler.start()
                self.profiler_started = True
            self.capture_started = time.perf_counter()
            self.capture_cpu_started = time.process_time()
            if self.monitor:
                self.sample_weights(parameters, "before_window")
                self.sample_memory("before_window")
        self.coordinate = {"update_index": index, "epoch": epoch, "batch": batch}
        self.captured.append(dict(self.coordinate))

    @contextmanager
    def region(self, name):
        if not self.active:
            yield
            return
        start = time.perf_counter()
        pair = None
        if self.monitor and self.device.type == "cuda":
            pair = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            pair[0].record(torch.cuda.current_stream(self.device))
        with self.label(name):
            yield
        if pair is not None:
            pair[1].record(torch.cuda.current_stream(self.device))
        row = {
            **self.coordinate,
            "region": name,
            "host_enqueue_s": time.perf_counter() - start,
        }
        self.phases.append(row)
        if pair is not None:
            self.events.append((row, pair))

    def label(self, name):
        if self.active and self.profile:
            return torch.profiler.record_function(f"snnlab::{name}")
        return nullcontext()

    def detail(self, name):
        return self.label(name) if self.spec.detailed_labels else nullcontext()

    def observe_inputs(self, inputs):
        if not self.active:
            return
        sample = next(iter(inputs.values()))
        if sample.ndim >= 2:
            self.coordinate.update(examples=sample.shape[1], steps=sample.shape[0])
            self.captured[-1].update(self.coordinate)
        if not self.monitor:
            return
        with torch.no_grad():
            for name, value in inputs.items():
                value = value.detach()
                if value.ndim == 3:
                    self._activity(
                        f"input/{name}",
                        value.count_nonzero(dim=(0, 1)),
                        value.shape[0] * value.shape[1],
                        (value == 0).all(dim=2).sum(),
                        (value == 0).all(dim=2).all(dim=1).sum(),
                        value.shape[0],
                    )

    def observe_spikes(self, spikes):
        if not self.active or not self.monitor:
            return
        with torch.no_grad():
            for name, value in spikes.items():
                if name not in self.spiking:
                    continue
                value = value.detach()
                self._activity(
                    f"population/{name}",
                    value.count_nonzero(dim=0),
                    value.shape[0],
                    (value == 0).all(dim=1).sum(),
                    (value == 0).all().to(torch.int64),
                    1,
                )

    def _activity(
        self, name, counts, example_steps, silent_example_steps, silent_steps, steps
    ):
        if name not in self.activity:
            self.activity[name] = {
                "counts": torch.zeros_like(counts),
                "example_steps": 0,
                "silent_example_steps": torch.zeros_like(silent_example_steps),
                "silent_steps": torch.zeros_like(silent_steps),
                "steps": 0,
            }
        row = self.activity[name]
        row["counts"].add_(counts)
        row["example_steps"] += example_steps
        row["silent_example_steps"].add_(silent_example_steps)
        row["silent_steps"].add_(silent_steps)
        row["steps"] += steps

    def sample_weights(self, parameters, phase):
        with torch.no_grad():
            for name, value in parameters.items():
                value = value.detach()
                row = {
                    "parameter": name,
                    "phase": phase,
                    "elements": value.numel(),
                    "nonzero": value.count_nonzero(),
                }
                if value.ndim == 2:
                    size = self.spec.block_size
                    padded = F.pad(
                        value != 0,
                        (0, -value.shape[1] % size, 0, -value.shape[0] % size),
                    )
                    blocks = padded.reshape(
                        padded.shape[0] // size, size, padded.shape[1] // size, size
                    )
                    row["nonzero_blocks"] = blocks.any(dim=3).any(dim=1).sum()
                    row["blocks"] = (padded.shape[0] // size) * (
                        padded.shape[1] // size
                    )
                self.weights.append(row)

    def observe_gradients(self, parameters):
        if not self.active or not self.monitor:
            return
        with torch.no_grad():
            for name, value in parameters.items():
                if value.grad is not None:
                    self.gradients.append(
                        {
                            **self.coordinate,
                            "parameter": name,
                            "elements": value.grad.numel(),
                            "nonzero": value.grad.detach().count_nonzero(),
                        }
                    )

    def sample_memory(self, phase):
        usage = (
            resource.getrusage(resource.RUSAGE_SELF) if resource is not None else None
        )
        self.memory.append(
            {
                "phase": phase,
                "process_peak_rss_bytes": usage.ru_maxrss
                * (1 if sys.platform == "darwin" else 1024)
                if usage
                else None,
                "cuda_allocated_bytes": torch.cuda.memory_allocated(self.device)
                if self.device.type == "cuda"
                else None,
                "cuda_reserved_bytes": torch.cuda.memory_reserved(self.device)
                if self.device.type == "cuda"
                else None,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(
                    self.device
                )
                if self.device.type == "cuda"
                else None,
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device)
                if self.device.type == "cuda"
                else None,
            }
        )

    def end_update(self, parameters, *, last=False):
        if not self.active:
            return
        if self.monitor:
            self.sample_memory(f"after_update_{self.coordinate['update_index']}")
        if last or len(self.captured) == self.spec.active_updates:
            self.stop_window(parameters)
        self.active = False

    def stop_window(self, parameters=None):
        if self.closed or self.capture_started is None:
            return
        if parameters is not None and self.monitor:
            self.sample_weights(parameters, "after_window")
        self._sync()
        self.window_wall_s = time.perf_counter() - self.capture_started
        self.cpu_s = time.process_time() - self.capture_cpu_started
        if self.profiler_started:
            self.profiler.stop()
            self.profiler_started = False
        for row, (start, end) in self.events:
            row["cuda_stream_elapsed_s"] = start.elapsed_time(end) / 1000
        self.events.clear()
        self.closed = True

    def finish(self, error=None):
        self.stop_window()
        self.active = False
        operators, timeline = [], timeline_metrics([], self.window_wall_s)
        if self.profiler is not None:
            trace_path = self.output / "trace.json"
            self.profiler.export_chrome_trace(str(trace_path))
            for event in self.profiler.key_averages(
                group_by_input_shape=self.spec.record_shapes
            ):
                operators.append(
                    {
                        "operator": event.key,
                        "calls": event.count,
                        "cpu_total_us": event.cpu_time_total,
                        "self_cpu_us": event.self_cpu_time_total,
                        "device_total_us": event.device_time_total,
                        "self_device_us": event.self_device_time_total,
                        "self_cpu_memory_bytes": event.self_cpu_memory_usage,
                        "self_device_memory_bytes": event.self_device_memory_usage,
                        "estimated_flops": event.flops if event.flops else None,
                        "flops_status": "estimated" if event.flops else "unavailable",
                        "input_shapes": str(event.input_shapes),
                    }
                )
            timeline = timeline_metrics(
                json.loads(trace_path.read_text()).get("traceEvents", []),
                self.window_wall_s,
            )

        def materialize(rows):
            return [
                {
                    key: value.item() if isinstance(value, torch.Tensor) else value
                    for key, value in row.items()
                }
                for row in rows
            ]

        weights, gradients = materialize(self.weights), materialize(self.gradients)
        for row in weights + gradients:
            row["zero_fraction"] = (
                1 - row["nonzero"] / row["elements"] if row["elements"] else None
            )
            if "blocks" in row:
                row["zero_block_fraction"] = 1 - row["nonzero_blocks"] / row["blocks"]
        activity = []
        for name, row in self.activity.items():
            counts = row["counts"].cpu().tolist()
            denominator = row["example_steps"]
            activity.append(
                {
                    "signal": name,
                    "density": sum(counts) / (denominator * len(counts)),
                    "mean_rate_hz": sum(counts)
                    / (denominator * len(counts))
                    * 1000
                    / self.dt_ms,
                    "per_neuron_rate_hz": [
                        value / denominator * 1000 / self.dt_ms for value in counts
                    ],
                    "silent_example_step_fraction": row["silent_example_steps"].item()
                    / denominator,
                    "globally_silent_step_fraction": row["silent_steps"].item()
                    / row["steps"],
                    "example_steps": denominator,
                }
            )
        config = asdict(self.spec)
        config["output"] = str(config["output"])
        unavailable = {
            name: {"status": "unavailable", "reason": reason}
            for name, reason in {
                "device_dram_bandwidth": "PyTorch Profiler does not expose DRAM byte counters",
                "sm_utilization": "requires hardware counters beyond this backend",
                "occupancy": "requires hardware counters beyond this backend",
                "cache_hit_rates": "requires hardware counters beyond this backend",
                "power_watts": "not provided by PyTorch Profiler",
                "saved_backward_tensor_bytes": "operator allocation attribution is available in profile mode; exact saved-tensor accounting is not collected",
            }.items()
        }
        summary = {
            "schema": "snnlab.tracing/v1",
            "configuration": config,
            "device": str(self.device),
            "torch_version": torch.__version__,
            "device_name": torch.cuda.get_device_name(self.device)
            if self.device.type == "cuda"
            else self.device.type,
            "cpu_threads": torch.get_num_threads(),
            "status": "failed"
            if error
            else "complete"
            if len(self.captured) == self.spec.active_updates
            else "truncated"
            if self.captured
            else "empty",
            "error": error,
            "captured_updates": self.captured,
            "scope": "contiguous call window spanning selected training updates; intervening evaluation is included in profile/wall time; activity statistics cover training only; all timings are instrumented",
            "window_wall_s": self.window_wall_s,
            "captured_examples": sum(row.get("examples", 0) for row in self.captured),
            "instrumented_examples_per_s": sum(
                row.get("examples", 0) for row in self.captured
            )
            / self.window_wall_s
            if self.window_wall_s
            else None,
            "cpu_process_s": self.cpu_s,
            "cpu_cores_used": self.cpu_s / self.window_wall_s
            if self.window_wall_s
            else None,
            "host_logical_cpus": os.cpu_count(),
            "cpu_scope": "process CPU seconds / capture wall seconds; includes worker threads; not normalized by reserved cloud CPUs",
            "memory_scope": "CUDA peaks reset at capture start; RSS peak is process lifetime, sampled at boundaries",
            "phase_scope": "host enqueue and current CUDA stream elapsed times overlap; nested phases must not be summed",
            "block_scope": "zero-padded edge tiles count as partial blocks; all parameters including frozen gain are listed",
            "measurement_status": {
                "activity": "measured" if activity else "unavailable",
                "weight_sparsity": "measured" if weights else "unavailable",
                "gradient_sparsity": "measured" if gradients else "unavailable",
                "cpu_peak_rss": "measured"
                if any(row["process_peak_rss_bytes"] is not None for row in self.memory)
                else "unavailable",
                "cuda_allocator": "measured"
                if self.memory and self.device.type == "cuda"
                else "unavailable",
                "host_phase_timing": "measured" if self.phases else "unavailable",
                "cuda_stream_timing": "measured"
                if any("cuda_stream_elapsed_s" in row for row in self.phases)
                else "unavailable",
                "cpu_activity": "derived" if self.captured else "unavailable",
                "kernel_coverage": timeline["kernel_coverage_status"],
                "copy_bandwidth": "derived"
                if any(row["GB_per_s"] is not None for row in timeline["copies"])
                else "unavailable",
                "operator_flops": "estimated"
                if any(row["estimated_flops"] is not None for row in operators)
                else "unavailable",
            },
            "unavailable": unavailable,
            "timeline": timeline,
            "activity": activity,
            "weights": weights,
            "gradients": gradients,
            "memory": self.memory,
            "phases": self.phases,
            "artifacts": [
                "summary.json",
                "phases.csv",
                "activity.csv",
                "weights.csv",
                "gradients.csv",
                "memory.csv",
            ]
            + (
                ["operators.csv", "copies.csv", "trace.json"]
                if self.profiler is not None
                else []
            ),
            "notes": [
                "Monitoring reductions and profiler events perturb timing and memory; compare with a separate uninstrumented baseline.",
                "FLOPs are PyTorch formula estimates for supported operators, not measured hardware FLOPs.",
                "Kernel coverage includes instrumentation kernels and is not a hardware utilization percentage.",
            ],
        }
        _write_json(self.output / "summary.json", summary)
        for name, rows in (
            ("phases", self.phases),
            ("activity", activity),
            ("weights", weights),
            ("gradients", gradients),
            ("memory", self.memory),
        ):
            _csv(self.output / f"{name}.csv", rows)
        if self.profiler is not None:
            _csv(self.output / "operators.csv", operators)
            _csv(self.output / "copies.csv", timeline["copies"])
        self.summary = summary
