"""Matched execution paths, cold/warm complete work and preparation provenance.

Run from the repository root, e.g. uv run python benchmarks/execution/benchmark.py.
This uses synthetic data and does not measure task accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch

from snnlab import lang
from snnlab.sim import (
    DatasetEncoder,
    DatasetSnapshotBinding,
    GraphExecutor,
    RecordingSpec,
    SignalRecording,
    plan_graph,
)
from snnlab.sim.diagnostics.tracing import timeline_metrics
from snnlab.sim.inputs.provider import DatasetProvider, iter_batches

ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def case_deadline(seconds):
    """Bound compiler/profiler work on POSIX hosts without hiding failures."""
    state = {"expired": False}

    def expire(*_):
        state["expired"] = True
        raise TimeoutError(f"benchmark case exceeded {seconds} seconds")

    previous = signal.signal(signal.SIGALRM, expire) if seconds else None
    if seconds:
        signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield state
    finally:
        if seconds:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def circuit(channels, cells, current):
    net = lang.Network("execution-benchmark", dt=1 * lang.ms)
    source = net.input(
        "events", shape=("time", "batch", channels), signal_type="spikes", unit="spike"
    )
    population = net.population(
        "cells",
        size=cells,
        neuron=lang.CUBA_LIF() if current else lang.COBA_LIF(tau_mem=20 * lang.ms),
    )
    synapse = (
        lang.ExponentialCurrent(tau=3 * lang.ms)
        if current
        else lang.AMPA(tau=3 * lang.ms)
    )
    net.connect(
        source,
        population.excitatory,
        name="drive",
        synapse=synapse,
        weight=lang.Constant(4 if current else 0.15),
    )
    net.connect(
        population.spikes,
        population.excitatory,
        name="recurrent",
        connection="recurrent",
        delay=2 * lang.ms,
        synapse=synapse,
        weight=lang.Constant(0.1),
        allow_autapses=False,
    )
    scores = lang.MeanVoltage(source=population.spikes, classes=2, name="readout")
    net.output("scores", scores)
    graph = lang.compile(net).graph
    return graph, tuple(row["id"] for row in graph["parameters"])


def source_provenance():
    paths = sorted(
        [
            *(ROOT / "src/snnlab").rglob("*.py"),
            *(ROOT / "benchmarks/execution").glob("*.py"),
        ]
    )
    # Modal receives a source snapshot rather than a Git checkout. Its launcher
    # writes the submitting checkout's identity beside that exact snapshot.
    snapshot = ROOT / ".execution-git.json"
    if snapshot.exists():
        identity = json.loads(snapshot.read_text())
    else:
        identity = {
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "git_dirty": bool(
                subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)
            ),
        }
    return {
        **identity,
        "sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
    }


def adam_options(execution):
    return {"default": {}, "foreach": {"foreach": True}, "fused": {"fused": True}}[
        execution
    ]


def autocast_context(policy):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if policy == "bfloat16"
        else nullcontext()
    )


def numerical_gate(
    graph,
    trainable,
    device,
    block_steps,
    dtype,
    steps,
    batch,
    adam_execution="default",
    autocast="off",
):
    """Use identical inputs and weights, independent of backend RNG streams."""
    generator = torch.Generator().manual_seed(71)
    inputs = (
        torch.rand(steps, batch, graph["inputs"][0]["shape"][-1], generator=generator)
        < 0.2
    ).to(device=device, dtype=dtype)
    tolerance = 3e-4 if dtype == torch.float32 and autocast == "off" else 5e-2
    results = []
    selected_precision = torch.get_float32_matmul_precision()
    try:
        for size, mode in ((1, None), (block_steps, "default")):
            torch.set_float32_matmul_precision(
                "highest" if mode is None else selected_precision
            )
            model = GraphExecutor(
                plan_graph(graph),
                seed=7,
                trainable_parameters=trainable,
                compile_mode=mode,
                compile_block_steps=size,
            ).to(device=device, dtype=dtype)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=0.001,
                **adam_options("default" if mode is None else adam_execution),
            )
            with autocast_context("off" if mode is None else autocast):
                result, _ = model._forward({"events": inputs}, diagnostics=False)
                loss = result.outputs["scores"].sum()
            loss.backward()
            gradients = {
                name: parameter.grad.detach().clone()
                for name, parameter in model.parameter_map().items()
                if parameter.grad is not None
            }
            optimizer.step()
            model.enforce_constraints()
            results.append((result, gradients, optimizer))
    finally:
        torch.set_float32_matmul_precision(selected_precision)
    expected, actual = results
    assert actual[1].keys() == expected[1].keys(), "gradient availability changed"
    for name, value in expected[0].outputs.items():
        torch.testing.assert_close(
            actual[0].outputs[name], value, atol=tolerance, rtol=tolerance
        )
    for name, value in expected[1].items():
        torch.testing.assert_close(
            actual[1][name], value, atol=tolerance, rtol=tolerance
        )
    for name, value in expected[0].model.parameter_map().items():
        torch.testing.assert_close(
            actual[0].model.parameter_map()[name], value, atol=tolerance, rtol=tolerance
        )
    for name, reference_parameter in expected[0].model.parameter_map().items():
        reference = expected[2].state[reference_parameter]
        candidate = actual[2].state[actual[0].model.parameter_map()[name]]
        assert candidate.keys() == reference.keys(), (
            "optimizer state availability changed"
        )
        for name, value in reference.items():
            torch.testing.assert_close(
                candidate[name],
                value.to(candidate[name].device),
                atol=tolerance,
                rtol=tolerance,
            )
    return {
        "passed": True,
        "atol": tolerance,
        "rtol": tolerance,
        "steps": steps,
        "batch": batch,
        "reference_matmul_precision": "highest",
        "candidate_matmul_precision": selected_precision,
        "adam_execution": adam_execution,
        "reference_adam_execution": "default",
        "reference_autocast": "off",
        "candidate_autocast": autocast,
        "optimizer_state_comparison": "Values compared on the candidate state device; fused Adam keeps its step counter on CUDA while default Adam may keep it on CPU.",
        "checks": ["outputs", "gradients", "Adam parameters", "Adam moments"],
    }


def measure(
    args,
    graph,
    trainable,
    binding,
    device,
    block_steps,
    compile_mode,
    preparation_device,
    batch_size,
):
    dtype = getattr(torch, args.dtype)
    # Provider construction/index validation is reported separately, not hidden.
    synchronize(device)
    started = time.perf_counter()
    provider = DatasetProvider(graph, binding, device=preparation_device)
    model = GraphExecutor(
        plan_graph(graph),
        seed=7,
        trainable_parameters=trainable,
        compile_mode=compile_mode,
        compile_block_steps=block_steps,
    ).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, **adam_options(args.adam_execution)
    )
    synchronize(device)
    setup_s = time.perf_counter() - started
    recording_bytes = 0

    def sink(block):
        nonlocal recording_bytes
        recording_bytes += block.values.numel() * block.values.element_size()

    recording = (
        RecordingSpec(
            (
                SignalRecording("cells.voltage"),
                SignalRecording("cells.spikes", kind="spike_events"),
            ),
            sink=sink,
            sink_id="execution-benchmark/v1",
            block_steps=32,
        )
        if args.record
        else None
    )
    last = None

    def epoch(*, profiling=False):
        nonlocal last
        sample_limit = (
            min(provider.sample_count, args.profile_batches * batch_size)
            if profiling
            else provider.sample_count
        )
        requests = (
            (
                torch.arange(begin, min(begin + batch_size, provider.sample_count)),
                "benchmark",
                0,
                None,
            )
            for begin in range(0, sample_limit, batch_size)
        )
        for _, tensors, _ in iter_batches(provider, requests, prefetch=args.prefetch):
            if profiling:
                tensors = {
                    name: value[: args.profile_steps] for name, value in tensors.items()
                }
            tensors = {
                name: value.to(device=device, dtype=dtype)
                for name, value in tensors.items()
            }
            optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(args.mode == "train"):
                # Keep reduction execution identical across paths for this
                # numerical benchmark. Online inference is measured separately
                # by native APIs; its blocks fallback remains explicit.
                with autocast_context(args.autocast):
                    last, _ = model._forward(
                        tensors,
                        diagnostics=False,
                        recording=recording,
                        retain_parameters=False,
                        retain_final_state=False,
                        retain_runtime_state=False,
                    )
                    if args.mode == "train":
                        loss = torch.nn.functional.cross_entropy(
                            last.outputs["scores"],
                            torch.zeros(
                                next(iter(tensors.values())).shape[1],
                                device=device,
                                dtype=torch.long,
                            ),
                        )
                if args.mode == "train":
                    loss.backward()
                    optimizer.step()
                    model.enforce_constraints()

    def timed_epoch():
        synchronize(device)
        started = time.perf_counter()
        epoch()
        synchronize(device)
        return time.perf_counter() - started

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    host_stats = getattr(torch.cuda.memory, "host_memory_stats", None)
    host_reset = getattr(torch.cuda.memory, "reset_peak_host_memory_stats", None)
    pinned_baseline = host_stats() if device.type == "cuda" and host_stats else None
    if device.type == "cuda" and host_reset:
        host_reset()
    cold_s = timed_epoch()
    warm = [timed_epoch() for _ in range(args.repeats)]
    profile = None
    if args.profile:
        # The bounded diagnostic can have a different tail shape from timing.
        # Warm that shape before counting dispatches/allocations.
        epoch(profiling=True)
        synchronize(device)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profile_started = time.perf_counter()
        with torch.profiler.profile(activities=activities, profile_memory=True) as prof:
            epoch(profiling=True)
            synchronize(device)
        profile_wall_s = time.perf_counter() - profile_started
        events = prof.key_averages()
        counts = {row.key: row.count for row in events}
        profile = {
            "simulation_steps_per_batch": min(args.profile_steps, provider.steps_count),
            "encoded_steps_per_batch": provider.steps_count,
            "max_batches": args.profile_batches,
            "aten_dispatches": sum(
                value for name, value in counts.items() if name.startswith("aten::")
            ),
            "host_scalar_reads": counts.get("aten::_local_scalar_dense", 0),
            "tensor_copy_calls": counts.get("aten::_to_copy", 0),
            "events": counts,
            "cuda_kernel_events": sum(
                event.device_type == torch.autograd.DeviceType.CUDA
                for event in prof.events()
            ),
            "note": "Dispatch counts are profiler observations, not a wall-time partition. MPS device kernels are not captured.",
        }
        if device.type == "cuda":
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "trace.json"
                prof.export_chrome_trace(str(path))
                timeline = timeline_metrics(
                    json.loads(path.read_text())["traceEvents"], profile_wall_s
                )
            copies = {}
            for copy in timeline.pop("copies"):
                row = copies.setdefault(
                    copy["name"],
                    {
                        "calls": 0,
                        "reported_bytes": 0,
                        "missing_byte_counts": 0,
                        "duration_us": 0,
                    },
                )
                row["calls"] += 1
                row["duration_us"] += copy["duration_us"]
                if copy["bytes"] is None:
                    row["missing_byte_counts"] += 1
                else:
                    row["reported_bytes"] += copy["bytes"]
            profile["cuda_timeline"] = {**timeline, "copies_by_name": copies}
            profile["host_waits"] = {
                row.key: {"calls": row.count, "self_cpu_us": row.self_cpu_time_total}
                for row in events
                if "Synchronize" in row.key or row.key == "aten::_local_scalar_dense"
            }
    pinned_final = host_stats() if device.type == "cuda" and host_stats else None
    pinned_issues = []
    if pinned_final:
        if any(
            value < 0 for key, value in pinned_final.items() if key.endswith(".freed")
        ):
            pinned_issues.append("backend exported negative freed counters")
        if pinned_final.get("active_bytes.current", 0) > pinned_final.get(
            "allocated_bytes.current", float("inf")
        ):
            pinned_issues.append("backend active bytes exceed allocator-owned bytes")
    return {
        "device": str(device),
        "dtype": args.dtype,
        "compile_mode": compile_mode,
        "adam_execution": args.adam_execution,
        "autocast": args.autocast,
        "requested_block_steps": block_steps,
        "batch_size": batch_size,
        "steps": provider.steps_count,
        "sample_count": provider.sample_count,
        "mode": args.mode,
        "recording": args.record,
        "prefetch": args.prefetch,
        "preparation_device": str(preparation_device),
        "encoding_policy": provider.protocol["batch_encoding"],
        "encoder": provider.protocol["dataset_binding"]["encoder"],
        "execution_path": last.metrics["execution_path"],
        "setup_s": setup_s,
        "cold_epoch_s": cold_s,
        "warm_epoch_s": warm,
        "warm_samples_per_s": provider.sample_count / statistics.median(warm),
        "warm_simulated_sample_seconds_per_s": provider.sample_count
        * provider.steps_count
        * 0.001
        / statistics.median(warm),
        "recording_bytes_all_epochs": recording_bytes,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "pinned_host_allocator_baseline": pinned_baseline,
        "pinned_host_allocator_stats": pinned_final,
        "pinned_host_allocator_stats_issues": pinned_issues,
        "pinned_host_memory_scope": "Case-interval peaks reset where supported; cached allocator blocks can predate this case. Null means the backend/API is unavailable.",
        "mps_current_allocated_bytes": torch.mps.current_allocated_memory()
        if device.type == "mps"
        else None,
        "process_lifetime_max_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "process_lifetime_max_rss_unit": "bytes" if sys.platform == "darwin" else "KiB",
        "profile": profile,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", default=["cpu"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[8])
    parser.add_argument("--steps", nargs="+", type=int, default=[33])
    parser.add_argument("--block-sizes", nargs="+", type=int, default=[1, 16, 32, 64])
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--cells", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prefetch", type=int, default=0)
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32", "float64"],
        default="float32",
    )
    parser.add_argument("--mode", choices=["train", "infer"], default="train")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-steps", type=int, default=128)
    parser.add_argument("--profile-batches", type=int, default=1)
    parser.add_argument(
        "--case-timeout",
        type=float,
        default=180,
        help="Per validation/timing case wall-time limit; 0 disables (POSIX).",
    )
    parser.add_argument("--coba", action="store_true")
    parser.add_argument(
        "--matmul-precision", choices=["highest", "high"], default="highest"
    )
    parser.add_argument(
        "--adam-execution", choices=["default", "foreach", "fused"], default="default"
    )
    parser.add_argument("--autocast", choices=["off", "bfloat16"], default="off")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/execution/benchmark.json")
    )
    args = parser.parse_args(argv)
    if args.autocast != "off" and (
        args.dtype != "float32"
        or any(torch.device(name).type != "cuda" for name in args.devices)
    ):
        parser.error("autocast experiments require CUDA and float32 model state")
    torch.set_float32_matmul_precision(args.matmul_precision)
    if (
        min(
            *args.batch_sizes,
            *args.steps,
            args.channels,
            args.cells,
            args.repeats,
            args.profile_steps,
            args.profile_batches,
        )
        < 1
    ):
        parser.error("sizes and repeats must be positive")
    if any(not 1 <= size <= 64 for size in args.block_sizes):
        parser.error("block sizes must be in [1, 64]")
    if args.case_timeout < 0:
        parser.error("case timeout must be nonnegative")
    graph, trainable = circuit(args.channels, args.cells, not args.coba)
    report = {
        "schema": "snnlab.execution-benchmark/v1",
        "status": "running",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "graph": graph,
        "synthetic": True,
        "provenance": source_provenance(),
        "software": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "threads": torch.get_num_threads(),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
        "hardware": {},
        "failures": [],
        "numerical_gates": [],
        "measurements": [],
        "limits": [
            "Synthetic workloads do not establish task accuracy.",
            "CPU and accelerator RNG streams differ; matched-input numerical gates are separate from pipeline timing.",
            "Cold epochs include compilation and lazy optimizer initialization; warmed samples include encoding, transfers, requested recording, forward, backward, optimizer and constraints.",
            "CPU RSS is a process-lifetime high-water mark, not an isolated case peak. MPS has no peak allocator metric here.",
            "Public CUDA Graph replay remains disabled; see the standalone fixed-shape training experiment.",
        ],
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def bounded(context, operation):
        state = None
        try:
            with case_deadline(args.case_timeout) as state:
                return operation()
        except Exception as error:
            if not state or not state["expired"]:
                report["status"] = "failed"
                report["failures"].append(
                    {
                        **context,
                        "status": "error",
                        "error_type": type(error).__name__,
                        "message": str(error)[:1000],
                    }
                )
                save()
                raise
            report["failures"].append(
                {
                    **context,
                    "status": "timeout",
                    "limit_s": args.case_timeout,
                    "error_type": type(error).__name__,
                }
            )
            save()
            print(json.dumps(report["failures"][-1]), flush=True)
            return None

    with tempfile.TemporaryDirectory(prefix="snnlab-execution-benchmark-") as temporary:
        for device_name in args.devices:
            device = torch.device(device_name)
            if device.type == "cuda":
                report["hardware"][str(device)] = torch.cuda.get_device_name(device)
            else:
                report["hardware"][str(device)] = (
                    platform.processor() or platform.machine()
                )
            for steps in args.steps:
                for batch in args.batch_sizes:
                    count = 2 * batch + max(1, batch // 2)
                    path = Path(temporary) / "data.npz"
                    rng = np.random.default_rng(71)
                    np.savez(
                        path,
                        features=rng.uniform(size=(count, args.channels)).astype(
                            np.float32
                        ),
                        labels=np.arange(count) % 2,
                    )
                    binding = DatasetSnapshotBinding(
                        path,
                        "events",
                        "synthetic-execution",
                        "benchmark",
                        DatasetEncoder(
                            "rate_poisson", duration_ms=steps, max_rate_hz=200, seed=19
                        ),
                    )
                    for block in args.block_sizes:
                        synchronize(device)
                        gate_started = time.perf_counter()
                        context = {
                            "device": str(device),
                            "steps": steps,
                            "batch": batch,
                            "block_steps": block,
                        }
                        gate = bounded(
                            {**context, "phase": "numerical_gate"},
                            lambda: numerical_gate(
                                graph,
                                trainable,
                                device,
                                block,
                                getattr(torch, args.dtype),
                                steps,
                                batch,
                                args.adam_execution,
                                args.autocast,
                            ),
                        )
                        if gate is None:
                            continue
                        synchronize(device)
                        gate["validation_and_first_execution_s"] = (
                            time.perf_counter() - gate_started
                        )
                        report["numerical_gates"].append(
                            {"device": str(device), "block_steps": block, **gate}
                        )
                        for prep in dict.fromkeys((torch.device("cpu"), device)):
                            modes = (None, "default") if block == 1 else ("default",)
                            for mode in modes:
                                row = bounded(
                                    {
                                        **context,
                                        "phase": "timing",
                                        "compile_mode": mode,
                                        "preparation_device": str(prep),
                                    },
                                    lambda: measure(
                                        args,
                                        graph,
                                        trainable,
                                        binding,
                                        device,
                                        block,
                                        mode,
                                        prep,
                                        batch,
                                    ),
                                )
                                if row is None:
                                    continue
                                report["measurements"].append(row)
                                save()
                                print(
                                    json.dumps(
                                        {
                                            key: row[key]
                                            for key in (
                                                "device",
                                                "steps",
                                                "batch_size",
                                                "compile_mode",
                                                "requested_block_steps",
                                                "preparation_device",
                                                "cold_epoch_s",
                                                "warm_samples_per_s",
                                            )
                                        }
                                    ),
                                    flush=True,
                                )
    report["status"] = "completed_with_timeouts" if report["failures"] else "complete"
    save()
    print(args.output.resolve())


if __name__ == "__main__":
    main()
