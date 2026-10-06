"""Voltage reductions retain their declared phase when masks are added."""

import copy
import math

import pytest
import torch

from snnlab import lang as snn
from snnlab.sim.execution import GraphExecutor, plan_graph
from tests.sim._bundle_builders import ping_classifier


def fixture(*, spiking=False):
    net = snn.Network("voltage_phases", dt=1 * snn.ms)
    events = net.input(
        "events", shape=("time", "batch", 1), signal_type="spikes", unit="spike"
    )
    valid = net.input("valid", shape=("time", "batch"), signal_type="mask")
    cell = net.population(
        "readout",
        size=1,
        neuron=snn.LeakyIntegrator(
            tau=2 * snn.ms, soft_reset_threshold=1.0, surrogate_slope=5.0
        ),
        spiking=spiking,
    )
    net.connect(
        events,
        cell.excitatory,
        name="drive",
        synapse=snn.LeakyIntegrator(tau=2 * snn.ms),
        weight=snn.Constant(3.0),
    )
    for phase, signal in (("pre", cell.pre_reset_voltage), ("post", cell.voltage)):
        net.output(f"{phase}_trace", signal)
        net.expose(signal, name=f"{phase}_diagnostic")
        for operation in ("mean", "sum"):
            for masked in (False, True):
                name = f"{phase}_{operation}_{'masked' if masked else 'full'}"
                value = snn.ops.reduce(
                    signal,
                    operation=operation,
                    over="time",
                    name=name,
                    mask=valid if masked else None,
                )
                net.output(f"{name}_result", value)
        final = net.operation(
            "select_final", signal, name=f"{phase}_final", shape=("batch", 1), unit="mV"
        )
        net.output(f"{phase}_final_result", final)
    return snn.compile(net)


def model(graph):
    return GraphExecutor(
        plan_graph(graph), seed=13, trainable_parameters=["drive.weight"]
    )


def inputs():
    return torch.tensor(
        [[[1.0], [1.0]], [[0.0], [1.0]], [[1.0], [0.0]], [[0.0], [1.0]], [[1.0], [0.0]]]
    )


def hand_traces(events):
    beta = math.exp(-0.5)
    voltage = torch.zeros(events.shape[1], 1)
    drive = torch.zeros_like(voltage)
    pre, post = [], []
    for event in events:
        drive = event * 3.0
        before = beta * voltage + (1 - beta) * drive
        voltage = before - (before > 1).float()
        pre.append(before)
        post.append(voltage)
    return torch.stack(pre), torch.stack(post)


@pytest.mark.parametrize("spiking", [False, True])
def test_declared_phases_match_hand_calculation_and_all_true_masks(spiking):
    bundle = fixture(spiking=spiking)
    events = inputs()
    result = model(bundle.graph)({"events": events, "valid": torch.ones(5, 2)})
    pre, post = hand_traces(events)
    assert torch.any(pre != post)  # The fixture actually crosses the reset threshold.
    for phase, expected in (("pre", pre), ("post", post)):
        torch.testing.assert_close(result.outputs[f"{phase}_trace"], expected)
        torch.testing.assert_close(result.diagnostics[f"{phase}_diagnostic"], expected)
        torch.testing.assert_close(
            result.outputs[f"{phase}_final_result"], expected[-1]
        )
        for operation in ("mean", "sum"):
            full = result.outputs[f"{phase}_{operation}_full_result"]
            masked = result.outputs[f"{phase}_{operation}_masked_result"]
            torch.testing.assert_close(full, masked, rtol=0, atol=0)
        assert result.numpy().outputs[f"{phase}_trace"].shape == (5, 2, 1)
    torch.testing.assert_close(result.runtime_state.voltages["readout"], post[-1])


def test_partial_masks_select_only_time_samples_of_the_same_phase():
    events = inputs()
    mask = torch.tensor([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0]], dtype=torch.float32)
    result = model(fixture().graph)({"events": events, "valid": mask})
    for phase in ("pre", "post"):
        trace = result.outputs[f"{phase}_trace"]
        expected_sum = (trace * mask[:, :, None]).sum(dim=0)
        expected_mean = expected_sum / mask.sum(dim=0)[:, None]
        torch.testing.assert_close(
            result.outputs[f"{phase}_sum_masked_result"], expected_sum
        )
        torch.testing.assert_close(
            result.outputs[f"{phase}_mean_masked_result"], expected_mean
        )


@pytest.mark.parametrize("phase", ["pre", "post"])
def test_all_true_masks_preserve_surrogate_gradients(phase):
    graph = fixture().graph
    module = model(graph)
    result = module({"events": inputs(), "valid": torch.ones(5, 2)})
    weight = module.parameter_map()["drive.weight"]
    unmasked = torch.autograd.grad(
        result.outputs[f"{phase}_mean_full_result"].sum(), weight, retain_graph=True
    )[0]
    masked = torch.autograd.grad(
        result.outputs[f"{phase}_mean_masked_result"].sum(), weight
    )[0]
    assert torch.isfinite(unmasked).all() and torch.any(unmasked != 0)
    torch.testing.assert_close(unmasked, masked, rtol=0, atol=0)


def test_continuation_and_disabled_diagnostics_preserve_both_phases():
    graph = fixture().graph
    events = inputs()
    whole = model(graph)({"events": events, "valid": torch.ones(5, 2)})
    continued = model(graph)
    first = continued(
        {"events": events[:2], "valid": torch.ones(2, 2)}, diagnostics=False
    )
    second = continued(
        {"events": events[2:], "valid": torch.ones(3, 2)},
        runtime_state=first.runtime_state,
        diagnostics=False,
    )
    assert first.diagnostics == second.diagnostics == {}
    for phase in ("pre", "post"):
        trace = torch.cat(
            [first.outputs[f"{phase}_trace"], second.outputs[f"{phase}_trace"]]
        )
        torch.testing.assert_close(
            trace, whole.outputs[f"{phase}_trace"], rtol=0, atol=0
        )
        totals = (
            first.outputs[f"{phase}_sum_full_result"]
            + second.outputs[f"{phase}_sum_full_result"]
        )
        torch.testing.assert_close(totals, whole.outputs[f"{phase}_sum_full_result"])
        mean = (
            2 * first.outputs[f"{phase}_mean_full_result"]
            + 3 * second.outputs[f"{phase}_mean_full_result"]
        ) / 5
        torch.testing.assert_close(mean, whole.outputs[f"{phase}_mean_full_result"])
        torch.testing.assert_close(
            second.outputs[f"{phase}_final_result"],
            whole.outputs[f"{phase}_final_result"],
        )


def test_legacy_graph_keeps_implicit_unmasked_mean_without_mutating_artifact():
    graph = fixture().graph
    graph.pop("voltage_sampling")
    original = copy.deepcopy(graph)
    events = inputs()
    result = model(graph)({"events": events, "valid": torch.ones(5, 2)})
    pre, post = hand_traces(events)
    # Historical mean alone took the implicit pre-reset route; masked mean and sums did not.
    torch.testing.assert_close(result.outputs["post_mean_full_result"], pre.mean(dim=0))
    torch.testing.assert_close(
        result.outputs["post_mean_masked_result"], post.mean(dim=0)
    )
    torch.testing.assert_close(result.outputs["post_sum_full_result"], post.sum(dim=0))
    assert graph == original


def test_mean_voltage_helper_and_bundle_roundtrip_select_pre_reset(tmp_path):
    bundle = ping_classifier()
    operation = bundle.graph["operations"][0]
    assert operation["sources"][0].endswith(".pre_reset_voltage")
    assert bundle.graph["voltage_sampling"] == "explicit"
    bundle.write(tmp_path / "bundle")
    assert snn.load_bundle(tmp_path / "bundle").graph == bundle.graph


def test_pre_reset_signal_is_limited_to_leaky_integrators_and_unknown_contract_fails():
    net = snn.Network("other_neuron")
    cell = net.population("E", size=1, neuron=snn.CUBA_LIF())
    with pytest.raises(AttributeError, match="leaky integrators"):
        _ = cell.pre_reset_voltage
    graph = fixture().graph
    graph["voltage_sampling"] = "unknown"
    assert any(
        "voltage_sampling" in error.message
        for error in snn.validate_graph(graph).errors
    )
    with pytest.raises(ValueError, match="voltage_sampling"):
        plan_graph(graph)




@pytest.mark.parametrize("phase", ["pre", "post"])
def test_composed_reductions_keep_phase_under_all_true_mask(phase):
    bundle = fixture()
    graph = bundle.graph
    source = f"readout.{'pre_reset_voltage' if phase == 'pre' else 'voltage'}"
    graph["operations"].append(
        {
            "id": "doubled",
            "kind": "linear",
            "sources": [source],
            "parameters": ["twice"],
            "shape": ["time", "batch", 1],
            "unit": "mV",
            "config": {},
        }
    )
    graph["parameters"].append(
        {
            "id": "twice",
            "shape": [1, 1],
            "unit": "1",
            "initializer": snn.Constant(2).json(),
        }
    )
    for suffix, sources, config in (
        ("full", ["doubled.value"], {}),
        ("masked", ["doubled.value", "valid.value"], {"mask": "valid.value"}),
    ):
        graph["operations"].append(
            {
                "id": f"composed_{suffix}",
                "kind": "reduce_mean",
                "sources": sources,
                "parameters": [],
                "shape": ["batch", 1],
                "unit": "mV",
                "config": config,
            }
        )
        graph["outputs"].append(
            {"id": f"composed_{suffix}", "signal": f"composed_{suffix}.value"}
        )
    result = model(graph)({"events": inputs(), "valid": torch.ones(5, 2)})
    expected = (2 * result.outputs[f"{phase}_trace"]).mean(dim=0)
    torch.testing.assert_close(result.outputs["composed_full"], expected)
    torch.testing.assert_close(result.outputs["composed_masked"], expected)
