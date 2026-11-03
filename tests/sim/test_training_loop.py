"""Canonical update ordering independent of graph construction details."""

from __future__ import annotations

import torch

from snnlab.sim.training.observations import GradientMeasurements
from snnlab.sim.training.policies import TrainingMetrics
from snnlab.sim.training.training_loop import (
    TrainingUpdateContext,
    run_training_update,
)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([2.0]))
        self.enforced_constraints = False
        self.forward_events = []

    def forward(self, values):
        return values * self.weight

    def enforce_constraints(self):
        self.enforced_constraints = True

    def _forward(self, inputs, **kwargs):
        self.forward_events.append("forward")
        return (
            type(
                "ForwardResult",
                (),
                {"outputs": {"prediction": inputs["events"].detach()}},
            )(),
            {"signal": inputs["events"]},
        )


class RecordingOptimizer(torch.optim.Optimizer):
    def __init__(self, parameter):
        super().__init__([parameter], {"lr": 0.0})
        self.events = []

    def step(self, closure=None):
        self.events.append("optimizer.step")
        return


class DisabledReplay:
    def __init__(self):
        self.events = []

    def run(self, inputs, targets, step):
        self.events.append("replay.run")
        return


def test_normal_update_calls_training_phases_in_order():
    model = FakeModel()
    optimizer = RecordingOptimizer(model.weight)
    replay = DisabledReplay()
    parameter_map = {"weight": model.weight}
    events = []
    loss = torch.tensor(3.0, requires_grad=True)

    def compute_loss(forward, signals, targets, protocol):
        events.append("compute_loss")
        return loss, {"objective[0]": loss}

    loss.backward = lambda: events.append("backward")
    context = TrainingUpdateContext(
        model=model,
        optimizer=optimizer,
        replay=replay,
        compute_loss=compute_loss,
        trace=None,
        parameter_map=parameter_map,
        optimizer_groups=tuple(optimizer.param_groups),
        training={"objectives": [], "gradient_clip": 1.0},
        training_recipe={"objectives": [], "gradient_clip": 1.0},
        metric_policy=TrainingMetrics(train="none", validation="none", initial=()),
        observation_spec=None,
        gradient_measurements=lambda: GradientMeasurements(()),
        input_protocol={},
        output_ids={},
        required_signals=(),
        device="cpu",
        diagnostics=False,
        precision="highest",
        targets={"label": torch.tensor([0])},
        dataset_inputs=None,
    )
    batches = iter(
        [(torch.tensor([0]), {"events": torch.tensor([[1.0]])}, {"draw": 0})]
    )

    outcome = run_training_update(
        context,
        update=0,
        update_count=1,
        scheduled=(0, 0, torch.tensor([0])),
        batches=batches,
    )

    assert [*model.forward_events, *events, *optimizer.events] == [
        "forward",
        "compute_loss",
        "backward",
        "optimizer.step",
    ]
    assert replay.events == ["replay.run"]
    assert optimizer.events == ["optimizer.step"]
    assert model.enforced_constraints is True
    assert outcome.online_metrics == {}
    assert outcome.online_count == 0
    assert outcome.before_clip_gradients == {}
    assert model.forward.__func__ is FakeModel.forward
