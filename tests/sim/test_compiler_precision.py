"""Actual Inductor must preserve intermediate reduced-dtype rounding."""

import pytest
import torch

from tests.sim import _compiler_precision as execution_benchmark


@pytest.mark.parametrize("block_steps", [1, 4])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS unavailable"
            ),
        ),
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_bfloat16_compilation_preserves_recurrent_gradients_and_adam(
    block_steps, device
):
    # With default Inductor cast elision this 128-step case changes every
    # drive-gradient element, with relative errors above 30% on CPU/CUDA.
    graph, trainable = execution_benchmark.circuit(32, 32, True)
    result = execution_benchmark.numerical_gate(
        graph, trainable, torch.device(device), block_steps, torch.bfloat16, 128, 32
    )
    assert result["passed"]
    assert result["compiler_precision_policy"] == "snnlab.eager-precision-casts/v1"


@pytest.mark.parametrize("version", ["2.10.0", "2.11.0"])
@pytest.mark.parametrize("block_steps", [1, 4])
def test_minimum_cpu_precision_fallback(block_steps, version, monkeypatch):
    monkeypatch.setattr(torch, "__version__", version)
    graph, trainable = execution_benchmark.circuit(32, 32, True)
    result = execution_benchmark.numerical_gate(
        graph, trainable, torch.device("cpu"), block_steps, torch.bfloat16, 128, 32
    )
    assert result["passed"]
    assert result["compiler_fallback"] == (
        f"PyTorch {'.'.join(version.split('.')[:2])} reduced-precision CPU uses eager tensor graphs"
    )


@pytest.mark.parametrize(
    "version,device,expected",
    [
        ("2.11.0+cu128", "cuda", "python"),
        ("2.11.1", "cuda", "python"),
        ("2.11.0", "cpu", "cpp"),
        ("2.12.0", "cuda", "cpp"),
        ("2.11.0", "mps", "python"),
    ],
)
def test_compiler_wrapper_compatibility_is_device_and_version_scoped(
    monkeypatch, version, device, expected
):
    from snnlab.sim.graph.kernels import compiler_wrapper_policy

    monkeypatch.setattr(torch, "__version__", version)
    policy = compiler_wrapper_policy(torch.device(device), torch.float32, "default")
    assert policy["selected"] == expected
    if device == "cuda" and version.startswith("2.11."):
        assert "CudaKernelParamCache" in policy["reason"]
    assert (
        compiler_wrapper_policy(torch.device(device), torch.float32, None)["selected"]
        == "eager"
    )


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_compiled_cuda_coba_dampened_surrogate_backward_matches_eager():
    from snnlab import lang, sim

    net = lang.Network("dampened-backward", dt=0.1 * lang.ms)
    drive = net.input(
        "drive", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
    )
    cells = net.population(
        "E",
        size=4,
        neuron=lang.COBA_LIF(tau_mem=20 * lang.ms, voltage_grad_dampen=1000),
    )
    net.connect(
        drive,
        cells.excitatory,
        name="input",
        synapse=lang.AMPA(tau=2 * lang.ms),
        weight=lang.Constant(0.3),
        initialization_scaling="direct",
    )
    activity = lang.reduce(cells.spikes, operation="sum", over="time", name="counts")
    net.output("activity", activity)
    graph = lang.compile(net).graph
    inputs = {"drive": torch.ones((20, 2, 2), device="cuda")}
    results = []
    for mode in (None, "default"):
        model = sim.GraphExecutor(
            sim.plan_graph(graph),
            trainable_parameters=("input.weight",),
            compile_mode=mode,
            precision="highest",
        ).to("cuda")
        result = model(inputs, diagnostics=False)
        result.outputs["activity"].sum().backward()
        gradient = model.parameter_map()["input.weight"].grad
        assert torch.isfinite(gradient).all() and gradient.abs().max() > 0
        results.append((result.outputs["activity"].detach(), gradient.detach()))
    for expected, actual in zip(*results, strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-4)
