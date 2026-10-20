"""Online inference reductions and explicitly retained recording blocks."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import deque
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

from snnlab import extensions as E
from snnlab.sim.artifacts.tensors import numpy_payload
from snnlab.sim.graph.validation import check_condition
from snnlab.sim.training.observations import evaluation_rng


def _step(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class MeasurementWindow:
    start_step: int
    end_step: int

    def validate(self):
        _step(self.start_step, "start_step")
        _step(self.end_step, "end_step")
        if self.end_step <= self.start_step:
            raise ValueError("measurement window must have positive duration")

    @classmethod
    def from_dict(cls, value):
        if set(value) != {"start_step", "end_step"}:
            raise ValueError("measurement requires start_step and end_step")
        result = cls(**value)
        result.validate()
        return result


@dataclass(frozen=True)
class SignalRecording:
    signal: str
    cells: tuple[int, ...] | None = None
    kind: str = "dense"


@dataclass(frozen=True)
class RecordingBlock:
    signal: str
    kind: str
    values: torch.Tensor
    start_step: int
    end_step: int
    dt_ms: float
    batch_offset: int = 0
    cells: tuple[int, ...] | None = None
    unit: str | None = None
    measurement_window: tuple[int, int] | None = None
    axes: tuple[int | str, ...] | None = None
    step_indices: tuple[int, ...] | None = None
    sample_ids: tuple[int | str, ...] | None = None
    batch_size: int | None = None
    selected_batch: int | None = None


class NPZRecordingSink:
    """Write bounded blocks and a digest manifest without retaining them in RAM."""

    identity = "snnlab.npz-recording-blocks/v1"

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if any(self.directory.iterdir()):
            raise ValueError("recording sink directory must be empty")
        self.index = 0

    def __call__(self, block: RecordingBlock):
        name = f"block-{self.index:08d}.npz"
        destination = self.directory / name
        values, storage = numpy_payload(block.values)
        np.savez_compressed(destination, values=values)
        metadata = {
            item.name: getattr(block, item.name) for item in fields(block) if item.name != "values"
        }
        metadata.update(
            schema=self.identity,
            file=name,
            shape=list(block.values.shape),
            dtype=str(block.values.dtype).removeprefix("torch."),
            **storage,
            sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        )
        with (self.directory / "manifest.jsonl").open("a") as handle:
            handle.write(json.dumps(metadata, sort_keys=True) + "\n")
        self.index += 1


@dataclass(frozen=True)
class RecordingSpec:
    signals: Sequence[SignalRecording] = field(default_factory=tuple)
    window: MeasurementWindow | None = None
    sink: Callable[[RecordingBlock], None] | None = None
    sink_id: str | None = None
    block_steps: int = 1

    @classmethod
    def from_dict(cls, value):
        if set(value) - {"signals", "window", "directory", "block_steps"}:
            raise ValueError("invalid recording fields")
        rows = tuple(SignalRecording(**row) for row in value.get("signals", ()))
        sink = NPZRecordingSink(value["directory"]) if "directory" in value else None
        return cls(
            rows,
            MeasurementWindow.from_dict(value["window"])
            if value.get("window")
            else None,
            sink,
            sink.identity if sink else None,
            value.get("block_steps", 1),
        )


def recording_identity(recording, *, axes):
    if recording is None:
        return None
    if not isinstance(recording, RecordingSpec):
        raise TypeError("recording must be RecordingSpec")
    if type(recording.block_steps) is not int or recording.block_steps < 1:
        raise ValueError("recording block_steps must be a positive integer")
    if recording.window:
        recording.window.validate()
    if recording.sink is not None and (
        not callable(recording.sink)
        or not isinstance(recording.sink_id, str)
        or not recording.sink_id
    ):
        raise ValueError("recording sinks require a callable and a stable sink_id")
    seen = set()
    for row in recording.signals:
        if not isinstance(row, SignalRecording):
            raise TypeError("recording signals must be SignalRecording")
        if row.signal in seen:
            raise ValueError("recording signals must be unique")
        seen.add(row.signal)
        shape = axes.get(row.signal, ())
        if (
            len(shape) != 3
            or shape[:2] != ("time", "batch")
            or type(shape[2]) is not int
        ):
            raise ValueError(
                f"recording requires a time/batch/cell signal: {row.signal}"
            )
        if row.kind not in {"dense", "spike_events"} or (
            row.kind == "spike_events" and not row.signal.endswith(".spikes")
        ):
            raise ValueError("spike_events recordings require population spikes")
        if row.cells is not None:
            if not row.cells or len(set(row.cells)) != len(row.cells):
                raise ValueError("recording cells must be non-empty and unique")
            for cell in row.cells:
                _step(cell, "cell index")
                if cell >= shape[2]:
                    raise ValueError("recording cell index exceeds signal size")
    return {
        "signals": [
            {
                "signal": row.signal,
                "cells": list(row.cells) if row.cells is not None else None,
                "kind": row.kind,
            }
            for row in recording.signals
        ],
        "window": asdict(recording.window) if recording.window else None,
        "sink_id": recording.sink_id,
        **(
            {"block_steps": recording.block_steps} if recording.block_steps != 1 else {}
        ),
    }


class Recorder:
    def __init__(
        self,
        spec,
        *,
        axes,
        start,
        steps,
        dt_ms,
        batch_offset=0,
        graph=None,
        dtype=torch.float32,
        device="cpu",
    ):
        self.spec = spec
        self.dtype = dtype
        self.axes = axes
        self.units = {}
        self.output_axes = {}
        if graph:
            self.output_axes = {row["id"]: axes[row["signal"]] for row in graph.get("outputs", ())}
            self.units.update(
                {
                    f"{row['id']}.value": row.get("unit")
                    for row in (*graph.get("inputs", ()), *graph.get("operations", ()))
                }
            )
            for signal in axes:
                owner, _, port = signal.partition(".")
                if port in {"voltage", "pre_reset_voltage"}:
                    self.units[signal] = "mV"
                elif port == "spikes":
                    self.units[signal] = "spike"
                elif port == "conductance":
                    self.units[signal] = "uS"
                elif port == "current":
                    self.units[signal] = "nA"
            for category, rows in (
                ("neuron", graph.get("populations", ())),
                ("synapse", graph.get("projections", ())),
            ):
                for row in rows:
                    self.units.update(
                        {
                            f"{row['id']}.{port}": unit
                            for port, unit in E.state_units(
                                category, row[category]
                            ).items()
                        }
                    )
            self.units.update(
                {
                    row["id"]: self.units.get(row["signal"])
                    for row in graph.get("outputs", ())
                }
            )
        self.identity = recording_identity(spec, axes=axes)
        self.start, self.end, self.dt_ms = start, start + steps, dt_ms
        self.batch_offset = batch_offset
        self.blocks = {}
        self.pending = {}
        self.transfers = deque()
        self.transfer_stream = None
        self.rows = {}
        self.cell_indices = {}
        self.selected_cells = {}
        if spec:
            self.rows = {row.signal: row for row in spec.signals}
            self.selected_cells = {
                row.signal: tuple(range(axes[row.signal][2]))
                if row.cells is None
                else tuple(row.cells)
                for row in spec.signals
            }
            self.cell_indices = {
                row.signal: torch.tensor(row.cells, device=device, dtype=torch.long)
                for row in spec.signals
                if row.cells is not None
            }

    def send(self, block):
        # User sinks cannot perturb the executor's random streams or tensors.
        if block.axes is None:
            axes = ("event", "coordinate") if block.kind == "spike_events" else self.output_axes.get(block.signal, self.axes.get(block.signal))
            if axes is None:
                raise ValueError(f"recording axis metadata is missing for {block.signal!r}")
            axes = tuple("cell" if isinstance(axis, int) else axis for axis in axes)
            size = block.values.shape[axes.index("batch")] if "batch" in axes else None
            block = replace(block, axes=axes,
                            step_indices=tuple(range(block.start_step, block.end_step)) if "time" in axes else None,
                            sample_ids=tuple(range(block.batch_offset, block.batch_offset + size)) if size is not None else block.sample_ids,
                            batch_size=block.batch_offset + size if size is not None else block.batch_size)
        with evaluation_rng():
            self.spec.sink(block)

    def observe(self, samples, step):
        if not self.spec or (
            self.spec.window
            and not self.spec.window.start_step <= step < self.spec.window.end_step
        ):
            return
        for name, row in self.rows.items():
            value = samples[name].detach()
            cells = self.selected_cells[name]
            if row.cells is not None:
                value = value.index_select(1, self.cell_indices[name])
            self.queue(name, row.kind, value, step, cells)

    def queue(self, name, kind, value, step, cells=None):
        # Snapshots are necessary because later steps and user sinks may mutate
        # their inputs. Event extraction happens on a CPU block, outside the loop.
        self.pending.setdefault(name, []).append(
            (step, value.detach().clone(), kind, cells)
        )
        if len(self.pending[name]) >= self.spec.block_steps:
            self.flush(name)

    def flush(self, name):
        rows = self.pending.pop(name, [])
        if not rows:
            return
        start, _, kind, cells = rows[0]
        end = rows[-1][0] + 1
        source = torch.stack([value for _, value, _, _ in rows])
        event = None
        if source.device.type == "cuda":
            if self.transfer_stream is None:
                self.transfer_stream = torch.cuda.Stream(device=source.device)
            self.transfer_stream.wait_stream(torch.cuda.current_stream(source.device))
            host = torch.empty_like(source, device="cpu", pin_memory=True)
            with torch.cuda.stream(self.transfer_stream):
                host.copy_(source, non_blocking=True)
                source.record_stream(self.transfer_stream)
                event = torch.cuda.Event()
                event.record(self.transfer_stream)
        else:
            host = source.cpu()
        self.transfers.append((name, kind, host, start, end, cells, event, source))
        # At most two in-flight host blocks; deliver completed older work while
        # the next chunk is copied. Sinks still run on the caller thread.
        if len(self.transfers) >= 2 or event is None:
            self.deliver()

    def deliver(self):
        name, kind, host, start, end, cells, event, source = self.transfers.popleft()
        if event is not None:
            event.synchronize()
        if kind == "spike_events":
            indices = host.nonzero()
            original = torch.tensor(cells, dtype=torch.int64)
            payload = torch.stack(
                (
                    indices[:, 0] + start,
                    indices[:, 1] + self.batch_offset,
                    original[indices[:, 2]],
                ),
                dim=1,
            )
        else:
            payload = host
        if self.spec.sink:
            self.send(
                RecordingBlock(
                    name,
                    kind,
                    payload,
                    start,
                    end,
                    self.dt_ms,
                    self.batch_offset,
                    cells,
                    self.units.get(name),
                    batch_size=self.batch_offset + host.shape[1],
                    sample_ids=tuple(range(self.batch_offset, self.batch_offset + host.shape[1])),
                )
            )
        else:
            self.blocks.setdefault(name, []).append(payload)

    def flush_all(self):
        for name in tuple(self.pending):
            self.flush(name)
        while self.transfers:
            self.deliver()

    def output_step(self, name, value, step):
        if self.spec and self.spec.sink:
            self.queue(name, "output", value, step)

    def output(self, name, value, measurement=None):
        self.flush_all()
        if self.spec and self.spec.sink:
            self.send(
                RecordingBlock(
                    name,
                    "output",
                    value.detach().cpu().clone(),
                    measurement.start_step if measurement else self.start,
                    max(measurement.start_step, min(self.end, measurement.end_step))
                    if measurement
                    else self.end,
                    self.dt_ms,
                    self.batch_offset,
                    unit=self.units.get(name),
                    measurement_window=(measurement.start_step, measurement.end_step)
                    if measurement
                    else None,
                )
            )

    def finish(self, batch_size):
        self.flush_all()
        return (
            {
                name: torch.cat(self.blocks[name])
                if self.blocks.get(name)
                else torch.empty((0, 3), dtype=torch.int64)
                if row.kind == "spike_events"
                else torch.empty(
                    (
                        0,
                        batch_size,
                        len(row.cells) if row.cells is not None else self.axes[name][2],
                    ),
                    dtype=self.dtype,
                )
                for name, row in self.rows.items()
            }
            if self.spec and not self.spec.sink
            else {}
        )

    def information(self, batch_size):
        """Describe retained coordinates without reading tensors or drawing RNG."""
        from snnlab.sim.artifacts.recordings import SignalInfo

        if not self.spec or self.spec.sink:
            return {}
        start = max(self.start, self.spec.window.start_step) if self.spec.window else self.start
        end = min(self.end, self.spec.window.end_step) if self.spec.window else self.end
        end = max(start, end)
        return {
            name: SignalInfo(
                axes=("event", "coordinate") if row.kind == "spike_events" else tuple("cell" if isinstance(axis, int) else axis for axis in self.axes[name]),
                kind=row.kind, unit=self.units.get(name), cell_ids=self.selected_cells[name],
                sample_ids=tuple(range(batch_size)),
                step_indices=None if row.kind == "spike_events" else tuple(range(start, end)),
                dt_ms=self.dt_ms, start_step=start, end_step=end,
                batch_size=batch_size, source_dtype="int64" if row.kind == "spike_events" else str(self.dtype).removeprefix("torch."),
            ) for name, row in self.rows.items()
        }


class OnlineReductions:
    def __init__(
        self,
        graph,
        axes,
        parameters,
        *,
        enabled,
        window,
        start,
        steps,
        saved,
        tensors,
        recording_signals=(),
        dtype=torch.float32,
        device=None,
    ):
        self.graph, self.axes, self.parameters = graph, axes, parameters
        self.window, self.start, self.steps = window, start, steps
        self.ops = {f"{row['id']}.value": row for row in graph.get("operations", ())}
        self.points = {}
        self.reductions = {}
        self.tensors = {}
        self.spec = None
        self.output_dtypes = {}
        self.duration_masks = {}
        self.dtype = dtype
        if window:
            if not isinstance(window, MeasurementWindow):
                raise TypeError("measurement must be MeasurementWindow")
            window.validate()
            if not enabled:
                raise ValueError(
                    "measurement windows require inference without autograd"
                )

        def pointwise(signal, selected):
            op = self.ops.get(signal)
            if op is None:
                return bool(axes.get(signal) and axes[signal][0] == "time")
            if op["kind"] not in {"linear", "pool2d"} or axes[signal][0] != "time":
                return False
            if not pointwise(op["sources"][0], selected):
                return False
            selected[signal] = op
            return True

        for signal in recording_signals:
            if not pointwise(signal, self.points):
                raise ValueError(
                    f"recording of this operation requires a dense history: {signal}"
                )
        if enabled:
            for signal, op in self.ops.items():
                if op["kind"] not in {"reduce_sum", "reduce_mean", "select_final"}:
                    continue
                selected = {}
                mask = op.get("config", {}).get("mask")
                if mask and mask in self.ops:
                    continue
                if not pointwise(op["sources"][0], selected):
                    continue
                self.points.update(selected)
                self.reductions[signal] = op
            for signal, op in self.ops.items():
                mask = op.get("config", {}).get("mask")
                if op["kind"] == "duration_normalise" and mask and mask not in self.ops:
                    self.duration_masks[signal] = mask
            if window:
                unsupported = [
                    op["id"]
                    for op in self.ops.values()
                    if op["kind"] in {"reduce_sum", "reduce_mean", "select_final"}
                    and f"{op['id']}.value" not in self.reductions
                ]
                if unsupported:
                    raise ValueError(
                        f"measurement window cannot stream these reductions: {unsupported}"
                    )
                for signal, op in self.ops.items():
                    if op["kind"] == "duration_normalise" and op.get("config", {}).get(
                        "mask"
                    ):
                        if op["config"]["mask"] in self.ops:
                            raise ValueError(
                                "measurement duration masks must be input signals"
                            )
                        self.duration_masks[signal] = op["config"]["mask"]
                self.spec = {
                    "window": asdict(window),
                    "reductions": list(self.reductions.values()),
                    "duration_masks": self.duration_masks,
                }
        if saved:
            if (
                saved.get("policy") != self.spec
                or saved.get("completed_steps") != start
            ):
                raise ValueError(
                    "runtime measurement policy/cursor does not match request"
                )
            expected_keys = {
                signal + suffix
                for signal in self.reductions
                for suffix in ("/value", "/count")
            }
            expected_keys.update(signal + "/duration" for signal in self.duration_masks)
            if set(tensors) != expected_keys:
                raise ValueError("runtime measurement tensor keys do not match policy")
            if (
                device is not None
                and torch.device(device).type == "mps"
                and any(value.dtype == torch.float64 for value in tensors.values())
            ):
                raise ValueError(
                    "runtime measurement float64 accumulators cannot be restored on MPS; continue on a float64-capable backend"
                )
            self.tensors = {
                name: value.detach().to(device=device).clone()
                for name, value in tensors.items()
            }
        elif self.spec and start:
            raise ValueError(
                "cannot attach a measurement window to an unaudited continuation"
            )
        self.initial_keys = set(self.tensors)
        self.used_keys = set()

    def history_operations(self, external):
        removed = set(self.reductions)
        changed = True
        while changed:
            changed = False
            for signal in self.points:
                if signal in removed or signal in external:
                    continue
                consumers = {
                    target for target, op in self.ops.items() if signal in op["sources"]
                }
                if consumers <= removed:
                    removed.add(signal)
                    changed = True
        return [op for signal, op in self.ops.items() if signal not in removed]

    def samples(self, values):
        from snnlab._shared.spatial import pool_tensor, signal_spatial_shape

        for signal, op in self.points.items():
            if op["kind"] == "pool2d":
                values[signal] = pool_tensor(
                    values[op["sources"][0]],
                    signal_spatial_shape(self.graph, op["sources"][0]),
                    op["config"],
                )
            else:
                values[signal] = (
                    values[op["sources"][0]] @ self.parameters[op["parameters"][0]]
                )
        return values

    def add(self, key, sample, *, final=False):
        self.used_keys.add(key)
        if key not in self.tensors:
            self.tensors[key] = torch.zeros_like(sample)
        value = self.tensors[key]
        if (
            value.shape != sample.shape
            or value.dtype != sample.dtype
            or value.device != sample.device
        ):
            raise ValueError("runtime measurement tensor shape/dtype/device mismatch")
        self.tensors[key] = sample.detach().clone() if final else value + sample

    def observe(self, values, step):
        active = (
            self.window is None or self.window.start_step <= step < self.window.end_step
        )
        for signal, op in self.reductions.items():
            source = op["sources"][0]
            owner, _, port = source.partition(".")
            if (
                "voltage_sampling" not in self.graph
                and op["kind"] == "reduce_mean"
                and not op.get("config", {}).get("mask")
                and port == "voltage"
                and f"{owner}.pre_reset_voltage" in values
            ):
                source = f"{owner}.pre_reset_voltage"
            sample = values[source]
            if op["kind"] == "reduce_mean" and not (
                sample.is_floating_point() or sample.is_complex()
            ):
                raise RuntimeError(
                    "mean requires a floating point or complex input dtype"
                )
            self.output_dtypes[signal] = (
                torch.int64
                if op["kind"] == "reduce_sum"
                and not (sample.is_floating_point() or sample.is_complex())
                else sample.dtype
            )
            mask_name = op.get("config", {}).get("mask")
            weight = (
                values[mask_name].to(sample.dtype)
                if mask_name
                else torch.ones(
                    sample.shape[0], device=sample.device, dtype=sample.dtype
                )
            )
            if weight.ndim != 1 or weight.shape[0] != sample.shape[0]:
                raise ValueError(
                    f"{op['id']}: valid-time mask must have shape [time, batch]"
                )
            weight = weight.reshape(sample.shape[0], *([1] * (sample.ndim - 1)))
            if op["kind"] == "reduce_sum" and not (
                sample.is_floating_point() or sample.is_complex()
            ):
                sample = sample.to(torch.int64)
                weight = weight.to(torch.int64)
            sequential = source.endswith(".pre_reset_voltage")
            if (
                not sequential
                and op["kind"] != "select_final"
                and sample.dtype
                in {torch.float16, torch.bfloat16, torch.float32, torch.float64}
            ):
                accumulator_dtype = (
                    torch.float32 if sample.device.type == "mps" else torch.float64
                )
                sample = sample.to(accumulator_dtype)
                weight = weight.to(accumulator_dtype)
            if op["kind"] == "select_final":
                self.add(
                    signal + "/value",
                    sample if active else torch.zeros_like(sample),
                    final=active,
                )
            else:
                self.add(
                    signal + "/value",
                    sample * weight if active else torch.zeros_like(sample),
                )
            self.add(signal + "/count", weight if active else torch.zeros_like(weight))
        for signal, mask in self.duration_masks.items():
            value = values[mask]
            if value.ndim != 1:
                raise ValueError("valid-time mask must have shape [time, batch]")
            self.add(
                signal + "/duration",
                value.to(self.dtype)
                if active
                else torch.zeros_like(value, dtype=self.dtype),
            )

    def finish(self, end):
        if self.initial_keys - self.used_keys:
            raise ValueError("runtime measurement contains unexpected tensors")
        result = {}
        complete = self.window is None or end >= self.window.end_step
        for signal, op in self.reductions.items():
            value = self.tensors[signal + "/value"]
            count = self.tensors[signal + "/count"]
            if op["kind"] == "reduce_mean":
                if complete:
                    check_condition(
                        (count > 0).all(),
                        f"{op['id']}: valid-time mask contains an empty reduction window",
                    )
                value = value / torch.where(count > 0, count, torch.ones_like(count))
            result[signal] = value.to(self.output_dtypes[signal])
        return result

    def state(self, end):
        return (
            {"policy": copy.deepcopy(self.spec), "completed_steps": end}
            if self.spec
            else {}
        )
