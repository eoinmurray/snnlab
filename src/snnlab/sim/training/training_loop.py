"""Canonical native training update, shared by normal and CUDA replay paths."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from snnlab.sim.graph import extensions as X
from snnlab.sim.graph.precision import execution_precision
from snnlab.sim.inputs.provider import iter_batches
from snnlab.sim.training.observations import EpochObservations, GradientMeasurements
from snnlab.sim.training.policies import TrainingMetrics
from snnlab.sim.training.replay import TrainingReplay

if TYPE_CHECKING:
    from snnlab.sim.graph.executor import GraphExecutor
    from snnlab.sim.requests.result import ExecutionResult


@dataclass(frozen=True)
class TrainingUpdateContext:
    """Immutable collaborators for one native update."""

    model: GraphExecutor
    optimizer: torch.optim.Optimizer
    replay: TrainingReplay
    compute_loss: Callable[..., tuple[torch.Tensor, dict[str, torch.Tensor]]]
    trace: Any
    parameter_map: Mapping[str, torch.Tensor]
    optimizer_groups: tuple[dict[str, Any], ...]
    training: Mapping[str, Any]
    training_recipe: Mapping[str, Any]
    metric_policy: TrainingMetrics
    observation_spec: EpochObservations | None
    gradient_measurements: Callable[[], GradientMeasurements]
    input_protocol: Mapping[str, Any]
    output_ids: Mapping[str, str]
    required_signals: tuple[str, ...]
    device: str
    diagnostics: bool
    precision: str
    targets: Mapping[str, torch.Tensor]
    dataset_inputs: Any
    microbatch_size: int | None = None


@dataclass
class TrainingUpdateOutcome:
    """Values owned by the outer trainer after one native update."""

    forward: ExecutionResult
    signals: Mapping[str, torch.Tensor]
    loss: torch.Tensor
    loss_components: dict[str, torch.Tensor]
    before_clip_gradients: dict[str, torch.Tensor]
    encoding_metadata: Any
    sample_indices: torch.Tensor
    epoch: int
    batch: int
    online_metrics: dict[str, torch.Tensor] = field(default_factory=dict)
    online_count: int = 0


def _prepare_batch(context, update, scheduled, batches, trace):
    """Fetch one scheduled batch and bind its targets."""
    if trace is not None:
        trace.begin_update(update, scheduled[0], scheduled[1], context.parameter_map)
    with trace.region("batch_prepare_wait") if trace is not None else nullcontext():
        _, batch_inputs, encoding_metadata = next(batches)
    if trace is not None:
        trace.observe_inputs(batch_inputs)
    epoch, batch, sample_indices = scheduled
    batch_targets = {
        name: value.index_select(0, sample_indices.to(value.device))
        for name, value in context.targets.items()
    }
    return batch_inputs, encoding_metadata, epoch, batch, sample_indices, batch_targets


def _forward_and_loss(
    context, batch_inputs, batch_targets, trace, *, accumulating=False
):
    """Run forward/loss directly, or return a captured replay result."""
    model = context.model

    def numerical_step(batch_inputs, batch_targets):
        forward, signals = model._forward(
            batch_inputs,
            **(
                {"initial_state": batch_inputs.initial_state}
                if getattr(batch_inputs, "initial_state", None) is not None
                else {}
            ),
            diagnostics=context.diagnostics,
            required_signals=context.required_signals,
            retain_parameters=False,
            retain_final_state=False,
            retain_runtime_state=False,
        )
        loss, components = context.compute_loss(
            forward, signals, batch_targets, context.input_protocol
        )
        with execution_precision(context.precision, next(model.parameters(), None)):
            loss.backward()
        return forward, signals, loss, components

    replayed = context.replay.run(batch_inputs, batch_targets, numerical_step)
    if replayed is not None:
        return replayed, True
    if not accumulating:
        context.optimizer.zero_grad(set_to_none=True)
    with trace.region("forward") if trace is not None else nullcontext():
        forward, signal_values = model._forward(
            batch_inputs,
            **(
                {"initial_state": batch_inputs.initial_state}
                if getattr(batch_inputs, "initial_state", None) is not None
                else {}
            ),
            diagnostics=context.diagnostics,
            required_signals=context.required_signals,
            retain_parameters=False,
            retain_final_state=False,
            retain_runtime_state=False,
        )
    with trace.region("loss") if trace is not None else nullcontext():
        loss, components = context.compute_loss(
            forward, signal_values, batch_targets, context.input_protocol
        )
    return (forward, signal_values, loss, components), False


def _accumulate_online_metrics(
    context, forward, loss, components, batch_targets, sample_indices
):
    """Collect pre-update online metrics without another forward pass."""
    if context.metric_policy.train != "online":
        return {}, 0
    count = len(sample_indices)
    values = {
        "loss": loss.detach() * count,
        **{
            f"component/{name}": value.detach() * count
            for name, value in components.items()
        },
    }
    online_sums = {}
    with torch.no_grad():
        for index, objective in enumerate(
            context.training_recipe.get("objectives", [])
        ):
            if X.classification(objective):
                prediction = forward.outputs[
                    context.output_ids[objective["prediction"]]
                ].detach()
                target = batch_targets[objective["target"]].long()
                values[f"ce/objective[{index}]"] = torch.nn.functional.cross_entropy(
                    prediction, target, reduction="sum"
                )
                values[f"correct/objective[{index}]"] = (
                    prediction.argmax(-1) == target
                ).sum()
        for name, value in values.items():
            value = value.to(
                dtype=torch.float64
                if torch.device(context.device).type != "mps"
                else torch.float32,
            )
            if name not in online_sums:
                online_sums[name] = torch.zeros_like(value)
            online_sums[name].add_(value)
    return online_sums, count


def _backward(context, loss, trace, replayed):
    """Backpropagate normal updates; replay already owns gradients."""
    with trace.region("backward") if trace is not None else nullcontext():
        if not replayed:
            with execution_precision(
                context.precision, next(context.model.parameters(), None)
            ):
                loss.backward()
    if trace is not None:
        trace.observe_gradients(context.parameter_map)


def _capture_and_clip_gradients(context, update, update_count):
    """Capture pre-clip gradients, then clip and record observations."""
    gradient_names = (
        set(context.parameter_map)
        if update == update_count - 1
        else set(context.observation_spec.gradient_norms)
        if context.observation_spec is not None
        else set()
    )
    before_clip = {
        name: parameter.grad.detach().clone()
        for name, parameter in context.parameter_map.items()
        if name in gradient_names and parameter.grad is not None
    }
    clip = context.training_recipe.get("gradient_clip")
    if clip is not None:
        torch.nn.utils.clip_grad_norm_(
            [
                parameter
                for group in context.optimizer_groups
                for parameter in group["params"]
            ],
            float(clip),
            error_if_nonfinite=True,
        )
    elif any(
        parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        for parameter in context.parameter_map.values()
    ):
        raise RuntimeError("nonfinite combined gradient")
    if context.observation_spec is not None:
        context.gradient_measurements().add(before_clip, context.parameter_map)
    return before_clip


def _optimizer_step_and_constraints(context, trace, update, update_count):
    """Apply the optimizer update and graph constraints as one boundary."""
    with trace.region("optimizer_constraints") if trace is not None else nullcontext():
        context.optimizer.step()
        enforce_constraints = getattr(context.model, "enforce_constraints", None)
        if enforce_constraints is not None:
            enforce_constraints()
    if trace is not None:
        trace.end_update(context.parameter_map, last=update + 1 == update_count)


def run_training_update(
    context: TrainingUpdateContext,
    *,
    update: int,
    update_count: int,
    scheduled: tuple[int, int, torch.Tensor],
    batches: iter_batches,
) -> TrainingUpdateOutcome:
    """Run one update while preserving the trainer's numerical boundaries."""
    if context.microbatch_size is not None:
        return _run_accumulated_update(
            context,
            update=update,
            update_count=update_count,
            scheduled=scheduled,
            batches=batches,
        )
    (
        batch_inputs,
        encoding_metadata,
        epoch,
        batch,
        sample_indices,
        batch_targets,
    ) = _prepare_batch(context, update, scheduled, batches, context.trace)
    (forward, signal_values, loss, components), replayed = _forward_and_loss(
        context, batch_inputs, batch_targets, context.trace
    )
    online_sums, online_count = _accumulate_online_metrics(
        context, forward, loss, components, batch_targets, sample_indices
    )
    _backward(context, loss, context.trace, replayed)
    before_clip = _capture_and_clip_gradients(context, update, update_count)
    _optimizer_step_and_constraints(context, context.trace, update, update_count)
    return TrainingUpdateOutcome(
        forward=forward,
        signals=signal_values,
        loss=loss,
        loss_components=components,
        before_clip_gradients=before_clip,
        encoding_metadata=encoding_metadata,
        sample_indices=sample_indices,
        epoch=epoch,
        batch=batch,
        online_metrics=online_sums,
        online_count=online_count,
    )


def _run_accumulated_update(context, *, update, update_count, scheduled, batches):
    """Weight each physical mean by its actual share of the effective batch."""
    epoch, batch, indices = scheduled
    context.optimizer.zero_grad(set_to_none=True)
    loss_total, components_total, online_sums = None, {}, {}
    encodings = []
    if context.trace is not None:
        context.trace.begin_update(update, epoch, batch, context.parameter_map)
    for local in indices.split(context.microbatch_size):
        with (
            context.trace.region("batch_prepare_wait")
            if context.trace is not None
            else nullcontext()
        ):
            _, inputs, metadata = next(batches)
        if context.trace is not None:
            context.trace.observe_inputs(inputs)
        targets = {
            name: value.index_select(0, local.to(value.device))
            for name, value in context.targets.items()
        }
        (forward, signals, loss, components), replayed = _forward_and_loss(
            context, inputs, targets, context.trace, accumulating=True
        )
        weight = len(local) / len(indices)
        with (
            context.trace.region("backward")
            if context.trace is not None
            else nullcontext()
        ):
            _backward(context, loss * weight, None, replayed)
        values, _ = _accumulate_online_metrics(
            context, forward, loss, components, targets, local
        )
        for name, value in values.items():
            online_sums[name] = online_sums.get(name, 0) + value
        loss_total = (
            loss.detach() * weight
            if loss_total is None
            else loss_total + loss.detach() * weight
        )
        for name, value in components.items():
            components_total[name] = (
                components_total.get(name, 0) + value.detach() * weight
            )
        encodings.append(metadata)
    if context.trace is not None:
        context.trace.observe_gradients(context.parameter_map)
    before_clip = _capture_and_clip_gradients(context, update, update_count)
    _optimizer_step_and_constraints(context, context.trace, update, update_count)
    return TrainingUpdateOutcome(
        forward,
        signals,
        loss_total,
        components_total,
        before_clip,
        {"microbatches": encodings},
        indices,
        epoch,
        batch,
        online_sums,
        len(indices) if context.metric_policy.train == "online" else 0,
    )
