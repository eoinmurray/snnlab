"""Configurable built-in SHD derivatives and compilation equivalence."""

import copy

import pytest
import torch

from snnlab import extensions, lang
from snnlab import lang as training
from snnlab.sim import GraphExecutor, plan_graph
from snnlab.sim.graph.dynamics import fast_sigmoid_spike
from snnlab.sim.requests.handlers import build
from snnlab.sim.requests.spec import ExecutionSpec
from tests.sim.test_gpu_efficiency import circuit


@pytest.mark.parametrize("inclusive", [True, False])
@pytest.mark.parametrize("scale", [0.0, 0.01, 1.0])
def test_derivative_and_threshold(inclusive, scale):
    value = torch.tensor(
        [-15.0, -0.1, 0.0, 0.1, 15.0], dtype=torch.float64, requires_grad=True
    )
    spike = fast_sigmoid_spike(value, 100 / 15, scale, inclusive)
    assert torch.equal(spike, ((value >= 0) if inclusive else (value > 0)).float())
    assert spike.dtype == value.dtype
    spike.sum().backward()
    torch.testing.assert_close(
        value.grad, scale * (100 / 15) / (1 + (100 / 15) * value.abs()) ** 2
    )


def bundle_with_surrogate(config):
    net = lang.Network("surrogate")
    events = net.input(
        "events", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
    )
    scores = lang.SpikeCount(source=events, classes=2, name="scores")
    net.output("class_scores", scores)
    recipe = lang.TrainSpec(
        objectives=[training.CrossEntropy(prediction=scores, target="label")],
        parameter_groups=[
            training.ParameterGroup(
                [p["id"] for p in net.parameters], name="all", lr=0.001
            )
        ],
        optimizer=training.AdamW(),
        surrogate=training.FastSigmoid(**config),
    )
    return lang.compile(net, training=recipe)


def test_serialization_build_and_default_identity(tmp_path):
    config = dict(slope=100 / 15, gradient_scale=0.01, threshold_inclusive=False)
    bundle = bundle_with_surrogate(config)
    bundle.write(tmp_path / "bundle")
    result = build(ExecutionSpec(kind="train", bundle=tmp_path / "bundle"))
    assert result.metrics["surrogate"] == {"kind": "fast_sigmoid", **config}
    value = torch.tensor([0.0], requires_grad=True)
    spike = result.model._spike_function(value)
    assert spike.item() == 0
    spike.backward()
    torch.testing.assert_close(value.grad, torch.tensor([1 / 15]))
    assert bundle_with_surrogate({}).training["surrogate"] == {
        "kind": "fast_sigmoid",
        "slope": 1.0,
    }


@pytest.mark.parametrize(
    "config",
    [
        {"gradient_scale": -1},
        {"gradient_scale": float("nan")},
        {"gradient_scale": float("inf")},
        {"gradient_scale": True},
        {"threshold_inclusive": 1},
        {"slope": 0},
    ],
)
def test_invalid_settings(config):
    with pytest.raises(ValueError):
        bundle_with_surrogate(config)
    graph, _ = circuit(lang.COBA_LIF)
    with pytest.raises(ValueError):
        GraphExecutor(plan_graph(graph), surrogate={"kind": "fast_sigmoid", **config})


@pytest.mark.parametrize(
    "constructor", [lang.COBA_LIF, lang.CUBA_LIF, lang.COBA_ALIF, lang.COBA_ADEX]
)
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_custom_eager_builtin_eager_compiled(constructor, device, monkeypatch):
    if device == "cpu":
        compile_fn = torch.compile
        monkeypatch.setattr(
            torch,
            "compile",
            lambda fn, **kw: compile_fn(fn, backend="eager", fullgraph=True),
        )
    monkeypatch.setattr(extensions, "_REGISTRY", {})
    graph, parameter = circuit(constructor)
    extensions.register_surrogate(
        "test.shd/v1",
        lambda value, config: (1 / 15) / (1 + (100 / 15) * value.abs()) ** 2,
    )
    configs = [
        (training.CustomSurrogate("test.shd/v1").json(), None),
        (
            training.FastSigmoid(
                slope=100 / 15, gradient_scale=0.01, threshold_inclusive=False
            ).json(),
            None,
        ),
        (
            training.FastSigmoid(
                slope=100 / 15, gradient_scale=0.01, threshold_inclusive=False
            ).json(),
            "default",
        ),
    ]
    values = torch.ones(13, 2, 2, device=device)
    results = []
    for surrogate, mode in configs:
        model = GraphExecutor(
            plan_graph(graph),
            seed=7,
            trainable_parameters=(parameter,),
            surrogate=surrogate,
            compile_mode=mode,
        ).to(device)
        full = model({"events": values}, diagnostics=False)
        (full.outputs["spikes"].sum() + full.outputs["voltage"].sum()).backward()
        gradient = model.parameter_map()[parameter].grad.clone()
        assert gradient.abs().sum() > 0
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.step()
        weights = {
            name: value.detach().clone()
            for name, value in model.parameter_map().items()
        }
        with torch.no_grad():
            whole = model({"events": values}, diagnostics=False)
            head = model({"events": values[:5]}, diagnostics=False)
            tail = model(
                {"events": values[5:]},
                diagnostics=False,
                runtime_state=head.runtime_state,
            )
        for name in ("voltage", "spikes"):
            torch.testing.assert_close(
                torch.cat((head.outputs[name], tail.outputs[name])),
                whole.outputs[name],
                atol=2e-5,
                rtol=2e-5,
            )
        results.append(
            (full.outputs, gradient, weights, copy.deepcopy(optimizer.state_dict()))
        )
    for result in results[1:]:
        for actual, expected in zip(result[:3], results[0][:3]):
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(result[3], results[0][3], atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_compiled_scaled_spikes_are_exact_binary(device):
    values = torch.cat(
        (torch.linspace(-30, 30, 1001, device=device), torch.zeros(1, device=device))
    ).requires_grad_()
    compiled = torch.compile(
        lambda value: fast_sigmoid_spike(value, 100 / 15, 0.01, False), fullgraph=True
    )
    spikes = compiled(values)
    assert torch.equal(spikes, (values > 0).float())
    spikes.sum().backward()
    torch.testing.assert_close(
        values.grad,
        (1 / 15) / (1 + (100 / 15) * values.abs()) ** 2,
        atol=1e-7,
        rtol=2e-5,
    )


@pytest.mark.parametrize("inclusive", [True, False])
def test_runtime_threshold_policy_and_provenance(inclusive):
    graph, _ = circuit(lang.COBA_LIF)
    surrogate = training.FastSigmoid(
        slope=100 / 15, gradient_scale=0.01, threshold_inclusive=inclusive
    ).json()
    model = GraphExecutor(plan_graph(graph), surrogate=surrogate)
    with torch.no_grad():
        result = model({"events": torch.ones(5, 2, 2)}, diagnostics=False)
        assert result.metrics["surrogate"]["threshold_inclusive"] is inclusive
        assert result.metrics["surrogate"]["gradient_scale"] == 0.01
        assert result.metrics["compile_mode"] == "default"
        if inclusive:
            assert (
                "builtin_spike_threshold_inclusive"
                not in result.runtime_state.compatibility
            )
        else:
            assert (
                result.runtime_state.compatibility["builtin_spike_threshold_inclusive"]
                is False
            )
        different = GraphExecutor(
            plan_graph(graph),
            surrogate=training.FastSigmoid(threshold_inclusive=not inclusive).json(),
        )
        with pytest.raises(ValueError, match="incompatible"):
            different(
                {"events": torch.ones(3, 2, 2)}, runtime_state=result.runtime_state
            )
        # Gradient amplitude changes do not change forward-state compatibility.
        same = GraphExecutor(
            plan_graph(graph),
            surrogate=training.FastSigmoid(
                gradient_scale=0.5, threshold_inclusive=inclusive
            ).json(),
        )
        same({"events": torch.ones(3, 2, 2)}, runtime_state=result.runtime_state)
