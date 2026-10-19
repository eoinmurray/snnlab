"""Validate custom constraints once per boundary; avoid unconstrained copies."""

import pytest
import torch

from snnlab import extensions as E
from snnlab import lang
from snnlab.sim import GraphExecutor, plan_graph

DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA unavailable"
        ),
    ),
]


def model(monkeypatch, callback):
    monkeypatch.setattr(E, "_REGISTRY", {})
    E.register_constraint("test.boundary/v1", callback)
    net = lang.Network("constraint-boundary", dt=1 * lang.ms)
    source = net.input(
        "events", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
    )
    cells = net.population("cells", size=2, neuron=lang.CUBA_LIF())
    for index in range(3):
        net.connect(
            source,
            cells.excitatory,
            name=f"drive{index}",
            synapse=lang.ExponentialCurrent(tau=3 * lang.ms),
            weight=lang.Constant(1),
            constraint=lang.CustomConstraint("test.boundary/v1") if index < 2 else None,
        )
    net.output("voltage", cells.voltage)
    return GraphExecutor(plan_graph(lang.compile(net).graph), compile_mode=None)


@pytest.mark.parametrize("device", DEVICES)
def test_custom_constraint_flags_share_one_host_read_and_only_constrained_weights_copy(
    monkeypatch,
    device,
):
    runner = model(monkeypatch, lambda value, config: value.clamp(-1, 1)).to(device)
    assert len(runner._constraints) == 2
    versions = {name: value._version for name, value in runner.parameter_map().items()}
    with torch.profiler.profile() as profile:
        runner.enforce_constraints()
    counts = {row.key: row.count for row in profile.key_averages()}
    assert counts["aten::_local_scalar_dense"] == 1
    for name, parameter in runner.parameter_map().items():
        assert parameter._version == versions[name] + int(name in runner._constraints)


@pytest.mark.parametrize("device", DEVICES)
def test_nonfinite_custom_constraint_still_raises_named_error(monkeypatch, device):
    invalid = False

    def constraint(value, config):
        return value * float("nan") if invalid else value

    runner = model(monkeypatch, constraint).to(device)
    invalid = True
    with pytest.raises(ValueError, match="test.boundary/v1 tensor must be finite"):
        runner.enforce_constraints()
