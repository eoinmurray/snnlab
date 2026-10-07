"""Online inference reductions and explicitly retained recording blocks."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

from snnlab import extensions as E
from snnlab.sim.epoch_observations import evaluation_rng


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
        np.savez_compressed(destination, values=block.values.numpy())
        metadata = {
            key: value for key, value in asdict(block).items() if key != "values"
        }
        metadata.update(
            file=name,
            shape=list(block.values.shape),
            dtype=str(block.values.numpy().dtype),
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

    @classmethod
    def from_dict(cls, value):
        if set(value) - {"signals", "window", "directory"}:
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
        )


def recording_identity(recording, *, axes):
    if recording is None:
        return None
    if not isinstance(recording, RecordingSpec):
        raise TypeError("recording must be RecordingSpec")
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
    ):
        self.spec = spec
        self.dtype = dtype
        self.axes = axes
        self.units = {}
        if graph:
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
        self.rows = {}
        if spec:
            self.rows = {row.signal: row for row in spec.signals}

    def send(self, block):
        # User sinks cannot perturb the executor's random streams or tensors.
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
            cells = (
                tuple(range(value.shape[1])) if row.cells is None else tuple(row.cells)
            )
            value = value[:, list(cells)]
            if row.kind == "spike_events":
                indices = value.nonzero()
                original = torch.tensor(cells, device=value.device, dtype=torch.int64)
                payload = torch.stack(
                    (
                        torch.full_like(indices[:, 0], step),
                        indices[:, 0] + self.batch_offset,
                        original[indices[:, 1]],
                    ),
                    dim=1,
                ).cpu()
            else:
                payload = value.unsqueeze(0).cpu().clone()
            if self.spec.sink:
                self.send(
                    RecordingBlock(
                        name,
                        row.kind,
                        payload,
                        step,
                        step + 1,
                        self.dt_ms,
                        self.batch_offset,
                        cells,
                        self.units.get(name),
                    )
                )
            else:
                self.blocks.setdefault(name, []).append(payload)

    def output_step(self, name, value, step):
        if self.spec and self.spec.sink:
            self.send(
                RecordingBlock(
                    name,
                    "output",
                    value.detach().unsqueeze(0).cpu().clone(),
                    step,
                    step + 1,
                    self.dt_ms,
                    self.batch_offset,
                    unit=self.units.get(name),
                )
            )

    def output(self, name, value, measurement=None):
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
            if op["kind"] != "linear" or axes[signal][0] != "time":
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
            self.tensors = {
                name: value.detach().clone() for name, value in tensors.items()
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
        for signal, op in self.points.items():
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
                and sample.device.type != "mps"
                and sample.dtype in {torch.float16, torch.float32, torch.float64}
            ):
                sample = sample.to(torch.float64)
                weight = weight.to(torch.float64)
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
                if complete and bool((count <= 0).any()):
                    raise ValueError(
                        f"{op['id']}: valid-time mask contains an empty reduction window"
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
