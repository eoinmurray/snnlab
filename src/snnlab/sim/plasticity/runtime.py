"""Plastic projections inside one graph forward pass.

The forward pass owns a :class:`PlasticRun` for every call that has plastic
projections. For each such projection it keeps one weight copy per trial and
the two traces of the rule, and it applies the rule's update step once per
time step, after the step's spikes are final:

* the presynaptic spikes are the ones arriving at the projection in this step,
  after its delay;
* the postsynaptic spikes are the target population's final spikes, after any
  intervention;
* transmission in step ``t`` uses the weights from the end of step ``t - 1``,
  because the drive is computed before the update.

The model's parameters and the bundle are never changed. Each call starts from
the model's weights, or from the weights and traces in ``runtime_state`` when
a run continues. Those live in ``custom_state`` under
``plasticity/<projection>/{weight,pre_trace,post_trace}``.

Entries that cannot change are masked: autapses the model excludes, and zero
weights of the model when the rule sets ``freeze_zero_weights``. A disabled
projection keeps its weights; its traces still track the spikes. The bounds
are checked before the run, on the entries that can change.

Three controls come from the caller. With plasticity off, as in training and
its evaluation passes, plastic projections run exactly like static ones. With
learning frozen, they transmit with their current per-trial weights, which do
not change, while their traces keep tracking spikes, as for a disabled
projection. Transmission scales multiply what a plastic projection transmits,
not the weights its rule learns from, so learning continues even at scale zero.

A continuing run needs the weights, both traces and the mask of entries that
can change; all four are saved in ``custom_state``. The saved mask wins over
the one the current model would give, as the saved weights do.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from snnlab._shared.plasticity import stdp_simple_from_json, unsupported_reasons
from snnlab.sim.plasticity import stdp_simple

STATE_PREFIX = "plasticity"
TRACE_PORTS = ("pre_trace", "post_trace")
STATE_NAMES = ("weight", "pre_trace", "post_trace", "mask")


def plastic_rows(graph: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return the projection rows that carry a plasticity rule."""
    return [row for row in graph.get("projections", []) if "plasticity" in row]


def validate_plastic_rows(graph: Mapping[str, Any]) -> None:
    """Reject invalid or unsupported plasticity before anything is built.

    Graph validation already reports these at compile and load time; this
    check keeps a hand-built graph from reaching the simulator.
    """
    for row in plastic_rows(graph):
        try:
            stdp_simple_from_json(row["plasticity"])
        except ValueError as error:
            raise ValueError(f"projection {row['id']!r}: {error}") from None
        reasons = unsupported_reasons(graph, row)
        if reasons:
            raise ValueError(
                f"projection {row['id']!r}: stdp_simple cannot apply to this "
                f"projection: {'; '.join(reasons)}"
            )


def plastic_signal_axes(graph: Mapping[str, Any], axes: Mapping[str, Any]) -> dict:
    """Recordable signals of plastic projections.

    ``<projection>.pre_trace`` and ``.post_trace`` are (time, batch, neurons);
    ``<projection>.weight`` is (time, batch, source, target) in the runtime
    layout, usually recorded every N steps.
    """
    sizes = {row["id"]: row["size"] for row in graph.get("populations", [])}
    shapes = {row["id"]: row["shape"] for row in graph.get("parameters", [])}
    result = {}
    for row in plastic_rows(graph):
        n_source = axes[row["source"]][-1]
        n_target = sizes[row["target"].partition(".")[0]]
        result[f"{row['id']}.pre_trace"] = ("time", "batch", n_source)
        result[f"{row['id']}.post_trace"] = ("time", "batch", n_target)
        # lang declares [target, source]; the runtime holds (source, target).
        source, target = reversed(shapes[row["parameters"][0]])
        result[f"{row['id']}.weight"] = ("time", "batch", source, target)
    return result


def state_key(projection_id: str, name: str) -> str:
    return f"{STATE_PREFIX}/{projection_id}/{name}"


@dataclass
class _Plastic:
    """One plastic projection during one call."""

    id: str
    parameter: str
    source: str
    delay_steps: int
    target: str
    rule: stdp_simple.STDPSimple
    one_to_one: bool
    shape: tuple[int, ...]  # the model parameter's runtime shape
    structure: torch.Tensor  # entries that can change: (source, target) or (n,)
    mask: torch.Tensor | None  # what the update step gets: None if all can change
    weights: torch.Tensor  # (batch, *shape)
    state: stdp_simple.STDPState


class PlasticRun:
    """Per-trial weights and traces of every plastic projection in one call."""

    def __init__(
        self,
        executor,
        *,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
        saved: Mapping[str, torch.Tensor] | None,
        enabled: bool = True,
        learning: bool = True,
        transmission_scales: Mapping[str, float] | None = None,
    ) -> None:
        if not isinstance(learning, bool):
            raise TypeError("learning must be True or False")
        self.dt_ms = executor.plan.dt_ms
        self.learning = learning
        self.projections: dict[str, _Plastic] = {}
        self.scales = _transmission_scales(transmission_scales)
        if not enabled:
            if self.scales:
                raise ValueError(
                    "transmission scales apply to plastic projections while "
                    "plasticity is on"
                )
            return
        rows = {row["id"]: row for row in plastic_rows(executor.plan.graph)}
        not_plastic = sorted(set(self.scales) - set(rows))
        if not_plastic:
            raise ValueError(
                f"transmission scales target projections that are not plastic: "
                f"{not_plastic}"
            )
        if not rows:
            return
        parameters = executor.parameter_map()
        for planned in executor.plan.projections:
            row = rows.get(planned.id)
            if row is None:
                continue
            spec = stdp_simple_from_json(row["plasticity"])
            freeze = spec.pop("freeze_zero_weights")
            rule = stdp_simple.STDPSimple(**spec)
            one_to_one = row.get("connectivity", {}).get("kind") == "one_to_one"
            with torch.no_grad():
                model = executor._mask_autapses(
                    planned.parameter, parameters[planned.parameter]
                ).detach()
                mask = (
                    executor._mask_autapses(planned.parameter, torch.ones_like(model))
                    != 0
                )
            if freeze:
                mask = mask & (model != 0)
            if not planned.enabled:
                mask = torch.zeros_like(mask)
            if one_to_one:
                mask = mask.reshape(-1)
            if saved is not None:
                mask = _restore(
                    saved, planned.id, "mask", tuple(mask.shape), torch.bool, device
                )
            n_source, n_target = (
                (model.numel(), model.numel()) if one_to_one else tuple(model.shape)
            )
            if saved is None:
                weights = model.unsqueeze(0).expand(batch, *model.shape).clone()
                state = stdp_simple.init_state(
                    rule,
                    batch=batch,
                    n_source=n_source,
                    n_target=n_target,
                    dtype=dtype,
                    device=device,
                )
            else:
                weights = _restore(
                    saved, planned.id, "weight", (batch, *model.shape), dtype, device
                )
                state = stdp_simple.STDPState(
                    pre_trace=_restore(
                        saved, planned.id, "pre_trace", (batch, n_source), dtype, device
                    ),
                    post_trace=_restore(
                        saved,
                        planned.id,
                        "post_trace",
                        (batch, n_target),
                        dtype,
                        device,
                    ),
                )
            plastic = _Plastic(
                id=planned.id,
                parameter=planned.parameter,
                source=planned.source,
                delay_steps=planned.delay_steps,
                target=planned.target.partition(".")[0],
                rule=rule,
                one_to_one=one_to_one,
                shape=tuple(model.shape),
                structure=mask,
                mask=None if bool(mask.all()) else mask,
                weights=weights,
                state=state,
            )
            _check_bounds(plastic)
            self.projections[planned.id] = plastic

    def __bool__(self) -> bool:
        return bool(self.projections)

    def drive(self, projection_id: str, source: torch.Tensor) -> torch.Tensor:
        """Transmit ``source`` (batch, source) with this trial's weights.

        A transmission scale multiplies the drive only; the weights the rule
        learns from stay unscaled.
        """
        plastic = self.projections[projection_id]
        weights = plastic.weights
        if plastic.one_to_one:
            drive = source * weights.reshape(weights.shape[0], -1)
        else:
            drive = torch.bmm(source.unsqueeze(1), weights).squeeze(1)
        scale = self.scales.get(projection_id)
        return drive if scale is None else drive * scale

    @torch.no_grad()
    def step(self, arrivals: Mapping[str, torch.Tensor], spikes) -> None:
        """Apply one update to every plastic projection.

        ``arrivals`` maps each projection to its presynaptic spikes at arrival;
        ``spikes`` maps each population to its final spikes of this step.
        """
        for plastic in self.projections.values():
            pre = arrivals[plastic.id]
            post = spikes[plastic.target]
            # Learning frozen: the weights stay, the traces keep tracking.
            mask = (
                plastic.mask if self.learning else torch.zeros_like(plastic.structure)
            )
            if plastic.one_to_one:
                batch = plastic.weights.shape[0]
                new, plastic.state = stdp_simple.step_one_to_one(
                    plastic.rule,
                    plastic.state,
                    plastic.weights.reshape(batch, -1),
                    pre,
                    post,
                    dt_ms=self.dt_ms,
                    mask=mask,
                )
                plastic.weights = new.reshape(plastic.weights.shape)
            else:
                plastic.weights, plastic.state = stdp_simple.step(
                    plastic.rule,
                    plastic.state,
                    plastic.weights,
                    pre,
                    post,
                    dt_ms=self.dt_ms,
                    mask=mask,
                )

    def signals(self) -> dict[str, torch.Tensor]:
        """Traces and weights after this step's update, as recordable signals."""
        result = {}
        for plastic in self.projections.values():
            result[f"{plastic.id}.pre_trace"] = plastic.state.pre_trace
            result[f"{plastic.id}.post_trace"] = plastic.state.post_trace
            result[f"{plastic.id}.weight"] = plastic.weights
        return result

    def custom_state(self) -> dict[str, torch.Tensor]:
        """What a continuing run needs: weights, both traces and the mask."""
        result = {}
        for plastic in self.projections.values():
            result[state_key(plastic.id, "weight")] = plastic.weights
            result[state_key(plastic.id, "pre_trace")] = plastic.state.pre_trace
            result[state_key(plastic.id, "post_trace")] = plastic.state.post_trace
            result[state_key(plastic.id, "mask")] = plastic.structure
        return result

    def parameters(self) -> dict[str, torch.Tensor]:
        """Final weights per trial, keyed by parameter id: ``(batch, *shape)``."""
        return {
            plastic.parameter: plastic.weights for plastic in self.projections.values()
        }


def _transmission_scales(scales: Mapping[str, float] | None) -> dict[str, float]:
    result = {}
    for projection_id, factor in (scales or {}).items():
        factor = float(factor)
        if not math.isfinite(factor) or factor < 0:
            raise ValueError(
                f"transmission scale of {projection_id!r} must be finite and "
                f"non-negative, got {factor}"
            )
        result[projection_id] = factor
    return result


def _restore(saved, projection_id, name, shape, dtype, device) -> torch.Tensor:
    key = state_key(projection_id, name)
    if key not in saved:
        raise ValueError(f"runtime state missing {key}")
    value = saved[key]
    if tuple(value.shape) != shape:
        raise ValueError(
            f"runtime state {key} shape expected {shape}, got {tuple(value.shape)}"
        )
    if value.dtype != dtype:
        raise ValueError(
            f"runtime state {key} dtype expected {dtype}, got {value.dtype}"
        )
    return value.detach().to(device).clone()


def _check_bounds(plastic: _Plastic) -> None:
    """Plastic entries must start within [w_min, w_max]; masked entries are exempt."""
    weights = plastic.weights
    if plastic.one_to_one:
        weights = weights.reshape(weights.shape[0], -1)
    changeable = weights if plastic.mask is None else weights[:, plastic.mask]
    if changeable.numel() == 0:
        return
    low, high = float(changeable.min()), float(changeable.max())
    rule = plastic.rule
    if low < rule.w_min or high > rule.w_max:
        raise ValueError(
            f"projection {plastic.id!r}: plastic weights must start within the "
            f"bounds [w_min, w_max] = [{rule.w_min}, {rule.w_max}]; found values "
            f"from {low} to {high}"
        )
