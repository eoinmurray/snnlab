"""Renderer-neutral recording arrays with per-signal coordinates."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from numbers import Integral
from pathlib import Path

import numpy as np
import torch

from snnlab.sim.artifacts.evidence import member_path, publish_directory, sha256_file
from snnlab.sim.artifacts.tensors import tensor_payload


@dataclass(frozen=True)
class SignalInfo:
    """Axes refer to array dimensions; event columns are step, batch, cell.

    ``step_indices`` are actual dense recording coordinates, never the full
    execution timeline. Event arrays carry their own integer step coordinates.
    """

    axes: tuple[int | str, ...]
    kind: str = "dense"
    unit: str | None = None
    cell_ids: tuple[int, ...] | None = None
    sample_ids: tuple[int | str, ...] | None = None
    step_indices: tuple[int, ...] | None = None
    dt_ms: float | None = None
    start_step: int | None = None
    end_step: int | None = None
    batch_size: int | None = None
    source_dtype: str | None = None
    array_dtype: str | None = None
    conversion: str | None = None
    selected_batch: int | None = None
    measurement_window: tuple[int, int] | None = None

    def __post_init__(self):
        if isinstance(self.axes, str):
            raise ValueError("recording axes must be a sequence of dimensions")
        for name in (
            "axes",
            "cell_ids",
            "sample_ids",
            "step_indices",
            "measurement_window",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, tuple(value))
        for name in ("start_step", "end_step", "batch_size", "selected_batch"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, Integral) or value < 0
            ):
                raise ValueError(f"recording {name} must be a nonnegative integer")
        if (
            self.start_step is not None
            and self.end_step is not None
            and self.end_step < self.start_step
        ):
            raise ValueError("recording end step precedes start step")
        if self.selected_batch is not None and (
            self.batch_size is None or self.selected_batch >= self.batch_size
        ):
            raise ValueError("selected batch exceeds the original recording batch size")
        if self.cell_ids is not None and (
            len(set(self.cell_ids)) != len(self.cell_ids)
            or any(
                isinstance(cell, bool) or not isinstance(cell, Integral) or cell < 0
                for cell in self.cell_ids
            )
        ):
            raise ValueError(
                "recording cell identities must be unique nonnegative integers"
            )
        if self.sample_ids is not None:
            expected = 1 if self.selected_batch is not None else self.batch_size
            if expected is not None and len(self.sample_ids) != expected:
                raise ValueError(
                    "recording sample identities disagree with batch metadata"
                )
            if len(set(self.sample_ids)) != len(self.sample_ids):
                raise ValueError("recording sample identities must be unique")
        if self.measurement_window is not None:
            window = self.measurement_window
            if (
                len(window) != 2
                or any(
                    isinstance(step, bool) or not isinstance(step, Integral) or step < 0
                    for step in window
                )
                or window[1] <= window[0]
            ):
                raise ValueError(
                    "measurement_window must have increasing nonnegative step bounds"
                )

    @property
    def time_ms(self):
        if self.step_indices is None or self.dt_ms is None:
            return None
        return np.asarray(self.step_indices, dtype=np.float64) * self.dt_ms


def numpy_copy(tensor):
    """Independent numeric arrays; finite bfloat16 values expand exactly."""
    value = tensor.detach().cpu()
    if value.dtype == torch.bfloat16:
        value = value.float()
    return value.numpy().copy()


def select_tensor(tensor, info, batch):
    if len(info.axes) != tensor.ndim:
        raise ValueError("recording axis metadata does not match tensor")
    if batch is None:
        return tensor, info
    if isinstance(batch, bool) or not isinstance(batch, Integral):
        raise TypeError("batch must be an integer or None")
    if info.batch_size is None or not 0 <= batch < info.batch_size:
        raise IndexError("recording batch is outside available sample coordinates")
    if info.selected_batch is not None:
        raise ValueError("recording has already selected a batch")
    sample_ids = (info.sample_ids[batch],) if info.sample_ids is not None else None
    if info.kind == "spike_events":
        selected = tensor[tensor[:, 1] == batch]
        axes = info.axes
    elif "batch" in info.axes:
        axis = info.axes.index("batch")
        selected = tensor.select(axis, int(batch))
        axes = info.axes[:axis] + info.axes[axis + 1 :]
    else:
        selected, axes = tensor, info.axes
    return selected, replace(
        info, axes=axes, sample_ids=sample_ids, selected_batch=int(batch)
    )


def _validate_time_coordinates(name, value, row):
    if "time" in row.axes:
        if (
            row.step_indices is None
            or len(row.step_indices) != value.shape[row.axes.index("time")]
        ):
            raise ValueError(f"{name}: time axis requires matching step coordinates")
        steps = np.asarray(row.step_indices)
        if any(
            isinstance(step, bool) or not isinstance(step, Integral) or step < 0
            for step in row.step_indices
        ) or (len(steps) > 1 and np.any(np.diff(steps) <= 0)):
            raise ValueError(f"{name}: step coordinates must increase strictly")
        if len(steps) and (
            (row.start_step is not None and steps[0] < row.start_step)
            or (row.end_step is not None and steps[-1] >= row.end_step)
        ):
            raise ValueError(f"{name}: step coordinates exceed recording window")


def _read_recording_blocks(root, rows):
    grouped = {}
    for row in rows:
        if (
            row.get("schema", "snnlab.npz-recording-blocks/v1")
            != "snnlab.npz-recording-blocks/v1"
        ):
            raise ValueError("unsupported recording schema")
        file = member_path(root, row["file"])
        if sha256_file(file) != row["sha256"]:
            raise ValueError("recording block digest mismatch")
        with np.load(file, allow_pickle=False) as archive:
            tensor = tensor_payload(archive["values"], row, label=row["signal"])
        axes = row.get("axes")
        if axes is None:
            if row["kind"] == "dense":
                axes = ("time", "batch", "cell")
            elif row["kind"] == "spike_events":
                axes = ("event", "coordinate")
            else:
                raise ValueError("historical output block lacks axis metadata")
        if (
            len(axes) != tensor.ndim
            or type(row.get("start_step")) is not int
            or type(row.get("end_step")) is not int
            or row["start_step"] < 0
            or row["end_step"] < row["start_step"]
        ):
            raise ValueError("recording block axes or time bounds are invalid")
        if (
            not isinstance(row.get("dt_ms"), (int, float))
            or isinstance(row["dt_ms"], bool)
            or not np.isfinite(row["dt_ms"])
            or row["dt_ms"] <= 0
        ):
            raise ValueError("recording block timestep must be finite and positive")
        grouped.setdefault(row["signal"], []).append((row, tensor, tuple(axes)))
    return grouped


def _assemble_dense_blocks(*, name, blocks, axes, first):
    # Blocks may tile time and batch independently (dataset inference).
    by_batch = {}
    for row, value, _ in blocks:
        by_batch.setdefault(row.get("batch_offset", 0), []).append((row, value))
    pieces, all_steps, batch_size = [], [], 0
    for offset, tile in sorted(by_batch.items()):
        tile.sort(key=lambda pair: pair[0]["start_step"])
        tile_steps = (
            tuple(
                step
                for row, _ in tile
                for step in (
                    row.get("step_indices")
                    if row.get("step_indices") is not None
                    else range(row["start_step"], row["end_step"])
                )
            )
            if "time" in axes
            else None
        )
        piece = (
            torch.cat([value for _, value in tile], dim=axes.index("time"))
            if "time" in axes
            else tile[0][1]
        )
        if "time" not in axes and len(tile) != 1:
            raise ValueError(f"{name}: duplicate non-temporal block")
        if "batch" in axes:
            if offset != batch_size:
                raise ValueError(f"{name}: missing or overlapping batch blocks")
            batch_size += piece.shape[axes.index("batch")]
        pieces.append(piece)
        all_steps.append(tile_steps)
    if any(item != all_steps[0] for item in all_steps):
        raise ValueError(f"{name}: batch blocks have different time coordinates")
    values = (
        torch.cat(pieces, dim=axes.index("batch")) if "batch" in axes else pieces[0]
    )
    if "batch" not in axes and len(pieces) > 1:
        raise ValueError(f"{name}: multiple batch tiles lack a declared batch axis")
    steps = all_steps[0]
    batch_size = first.get("batch_size") if "batch" not in axes else batch_size
    return values, steps, batch_size


def _validate_event_coordinates(name, value, row):
    if row.kind == "spike_events":
        if row.axes != ("event", "coordinate"):
            raise ValueError(f"{name}: unsupported event axes")
        if (
            value.ndim != 2
            or value.shape[1] != 3
            or not np.issubdtype(value.dtype, np.integer)
        ):
            raise ValueError(f"{name}: events require integer step/batch/cell columns")
        if value.size and (
            np.any(value < 0)
            or (row.start_step is not None and np.any(value[:, 0] < row.start_step))
            or (row.end_step is not None and np.any(value[:, 0] >= row.end_step))
        ):
            raise ValueError(f"{name}: event coordinates exceed recording window")
        if (
            value.size
            and row.batch_size is not None
            and np.any(value[:, 1] >= row.batch_size)
        ):
            raise ValueError(f"{name}: event sample coordinates exceed batch size")
        if (
            row.selected_batch is not None
            and value.size
            and np.any(value[:, 1] != row.selected_batch)
        ):
            raise ValueError(
                f"{name}: event coordinates disagree with the selected sample"
            )
        if (
            row.cell_ids is not None
            and value.size
            and not np.isin(value[:, 2], row.cell_ids).all()
        ):
            raise ValueError(f"{name}: unknown event cell identity")
    elif row.kind not in {"dense", "output"}:
        raise ValueError(f"{name}: unsupported recording kind")


class RecordingData:
    """Independent arrays and truthful metadata shared by readers and plotters."""

    def __init__(
        self, signals: Mapping[str, np.ndarray], info: Mapping[str, SignalInfo]
    ):
        self.signals = {
            name: np.asarray(value).copy() for name, value in signals.items()
        }
        self.info = dict(info)
        self.validate()

    def validate(self):
        if set(self.signals) != set(self.info):
            raise ValueError(
                "recording signals and metadata must have matching identities"
            )
        for name, value in self.signals.items():
            row = self.info[name]
            if (
                not isinstance(row, SignalInfo)
                or value.dtype.kind not in {"b", "i", "u", "f", "c"}
                or len(row.axes) != value.ndim
            ):
                raise ValueError(f"{name}: unsupported dtype or inconsistent axes")
            if len({axis for axis in row.axes if isinstance(axis, str)}) != len(
                [axis for axis in row.axes if isinstance(axis, str)]
            ):
                raise ValueError(f"{name}: named axes must be unique")
            if row.dt_ms is not None and (not np.isfinite(row.dt_ms) or row.dt_ms <= 0):
                raise ValueError(f"{name}: invalid timestep")
            _validate_event_coordinates(name=name, value=value, row=row)

            if row.selected_batch is not None and "batch" in row.axes:
                raise ValueError(
                    f"{name}: selected recordings cannot retain a batch axis"
                )
            if "time" not in row.axes and row.step_indices is not None:
                raise ValueError(f"{name}: step coordinates require a time axis")
            _validate_time_coordinates(name=name, value=value, row=row)

            if (
                "cell" in row.axes
                and row.cell_ids is not None
                and len(row.cell_ids) != value.shape[row.axes.index("cell")]
            ):
                raise ValueError(f"{name}: cell identities do not match the cell axis")
            if (
                "batch" in row.axes
                and row.batch_size != value.shape[row.axes.index("batch")]
            ):
                raise ValueError(f"{name}: batch size does not match its axis")

    def select_batch(self, batch):
        signals, info = {}, {}
        for name, value in self.signals.items():
            selected, row = select_tensor(
                torch.from_numpy(value), self.info[name], batch
            )
            signals[name], info[name] = numpy_copy(selected), row
        return RecordingData(signals, info)

    @classmethod
    def read(cls, path):
        """Read/verify native NPZ sink blocks, including historical dense blocks."""
        root = Path(path)
        rows = [
            json.loads(line)
            for line in (root / "manifest.jsonl").read_text().splitlines()
            if line.strip()
        ]
        grouped = _read_recording_blocks(root=root, rows=rows)

        signals, metadata = {}, {}
        for name, blocks in grouped.items():
            first, _, axes = blocks[0]
            if any(
                axis != axes
                or row["kind"] != first["kind"]
                or row.get("unit") != first.get("unit")
                or row["dt_ms"] != first["dt_ms"]
                or row.get("cells") != first.get("cells")
                or row["dtype"] != first["dtype"]
                or row.get("selected_batch") != first.get("selected_batch")
                or row.get("measurement_window") != first.get("measurement_window")
                for row, _, axis in blocks
            ):
                raise ValueError(f"{name}: inconsistent recording block metadata")
            identities = {}
            for row, _, _ in blocks:
                offset = row.get("batch_offset", 0)
                if type(offset) is not int or offset < 0:
                    raise ValueError(f"{name}: invalid recording batch offset")
                ids = row.get("sample_ids")
                if offset in identities and identities[offset] != ids:
                    raise ValueError(
                        f"{name}: sample identities change between time blocks"
                    )
                identities[offset] = ids
            kind = first["kind"]
            steps = None
            if kind == "spike_events":
                values = torch.cat([value for _, value, _ in blocks], dim=0)
                sizes = [row.get("batch_size") for row, _, _ in blocks]
                batch_size = (
                    max(sizes) if all(size is not None for size in sizes) else None
                )
            else:
                values, steps, batch_size = _assemble_dense_blocks(
                    name=name, blocks=blocks, axes=axes, first=first
                )

            source_dtype = str(values.dtype).removeprefix("torch.")
            array = numpy_copy(values)
            sample_ids = first.get("sample_ids")
            if ("batch" in axes or kind == "spike_events") and len(blocks) > 1:
                # Different time blocks repeat their sample IDs.
                by_offset = {
                    row.get("batch_offset", 0): row.get("sample_ids")
                    for row, _, _ in blocks
                }
                sample_ids = (
                    tuple(item for _, ids in sorted(by_offset.items()) for item in ids)
                    if all(ids is not None for ids in by_offset.values())
                    else None
                )
            info = SignalInfo(
                axes,
                kind,
                first.get("unit"),
                tuple(first["cells"]) if first.get("cells") is not None else None,
                tuple(sample_ids) if sample_ids is not None else None,
                steps,
                first["dt_ms"],
                min(row["start_step"] for row, _, _ in blocks),
                max(row["end_step"] for row, _, _ in blocks),
                batch_size,
                source_dtype,
                str(array.dtype),
                "bfloat16_to_float32" if source_dtype == "bfloat16" else None,
                first.get("selected_batch"),
                first.get("measurement_window"),
            )
            signals[name], metadata[name] = array, info
        return cls(signals, metadata)


def write_recording(path, tensors, info, *, signals=None, batch=None, overwrite=False):
    """Write requested retained tensors through the native recording sink."""
    from snnlab.sim.runtime.recording import NPZRecordingSink, RecordingBlock

    names = tuple(tensors) if signals is None else tuple(signals)
    if len(set(names)) != len(names):
        raise ValueError("recording signal names must be unique")
    if set(names) - set(tensors):
        raise ValueError(
            f"requested recordings are unavailable: {sorted(set(names) - set(tensors))}"
        )
    selected = {}
    for name in names:
        if name not in info:
            raise ValueError(f"recording metadata is missing for {name!r}")
        selected[name] = select_tensor(tensors[name], info[name], batch)

    def write(root):
        sink = NPZRecordingSink(root)
        # An empty export is still a readable recording archive.
        (root / "manifest.jsonl").touch()
        for name, (tensor, row) in selected.items():
            if row.dt_ms is None or row.start_step is None or row.end_step is None:
                raise ValueError(f"{name}: recording export requires timebase metadata")
            sink(
                RecordingBlock(
                    name,
                    row.kind,
                    tensor,
                    row.start_step,
                    row.end_step,
                    row.dt_ms,
                    cells=row.cell_ids,
                    unit=row.unit,
                    axes=row.axes,
                    step_indices=row.step_indices,
                    sample_ids=row.sample_ids,
                    batch_size=row.batch_size,
                    selected_batch=row.selected_batch,
                    measurement_window=row.measurement_window,
                )
            )

    return publish_directory(path, write, overwrite=overwrite)
