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


def run_training_update(
    context: TrainingUpdateContext,
    *,
    update: int,
    update_count: int,
    scheduled: tuple[int, int, torch.Tensor],
    batches: iter_batches,
) -> TrainingUpdateOutcome:
    """Run one update while preserving the trainer's numerical boundaries."""
    model = context.model
    trace = context.trace
    if trace is not None:
        trace.begin_update(
            update,
            scheduled[0],
            scheduled[1],
            context.parameter_map,
        )
    with trace.region("batch_prepare_wait") if trace is not None else nullcontext():
        _, batch_inputs, encoding_metadata = next(batches)
    if trace is not None:
        trace.observe_inputs(batch_inputs)
    epoch, batch, sample_indices = scheduled
    batch_targets = {
        name: value.index_select(0, sample_indices.to(value.device))
        for name, value in context.targets.items()
    }

    def numerical_step(batch_inputs, batch_targets):
        forward, signals = model._forward(
            batch_inputs,
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
        forward, signal_values, loss, components = replayed
    else:
        context.optimizer.zero_grad(set_to_none=True)
        with trace.region("forward") if trace is not None else nullcontext():
            forward, signal_values = model._forward(
                batch_inputs,
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

    online_sums = {}
    online_count = 0
    if context.metric_policy.train == "online":
        count = len(sample_indices)
        values = {
            "loss": loss.detach() * count,
            **{
                f"component/{name}": value.detach() * count
                for name, value in components.items()
            },
        }
        with torch.no_grad():
            for index, objective in enumerate(
                context.training_recipe.get("objectives", [])
            ):
                if X.classification(objective):
                    prediction = forward.outputs[
                        context.output_ids[objective["prediction"]]
                    ].detach()
                    target = batch_targets[objective["target"]].long()
                    values[f"ce/objective[{index}]"] = (
                        torch.nn.functional.cross_entropy(
                            prediction, target, reduction="sum"
                        )
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
        online_count += count
    with trace.region("backward") if trace is not None else nullcontext():
        if replayed is None:
            with execution_precision(context.precision, next(model.parameters(), None)):
                loss.backward()
    if trace is not None:
        trace.observe_gradients(context.parameter_map)
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
        )
    if context.observation_spec is not None:
        context.gradient_measurements().add(before_clip, context.parameter_map)
    with trace.region("optimizer_constraints") if trace is not None else nullcontext():
        context.optimizer.step()
        model.enforce_constraints()
    if trace is not None:
        trace.end_update(context.parameter_map, last=update + 1 == update_count)
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
