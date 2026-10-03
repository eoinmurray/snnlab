"""Actual Inductor execution of the public network's production step body."""

import pytest
import torch

from snnlab.sim import models as M
from snnlab.sim.config import build_net, set_sim_dt


@pytest.mark.slow
@pytest.mark.parametrize(
    "device_name",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.accelerator,
                pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA is unavailable"
                ),
            ],
        ),
    ],
)
@pytest.mark.parametrize("dt", [0.05, 0.1, 0.2, 0.3, 0.6])
def test_inductor_matches_eager_refractory_forward_and_backward(
    monkeypatch, dt, device_name
):
    monkeypatch.setattr(M, "N_IN", 3)
    monkeypatch.setattr(M, "N_OUT", 3)
    set_sim_dt(dt, 12.0)
    torch.manual_seed(73)
    kwargs = dict(
        device=torch.device(device_name),
        hidden_sizes=[8],
        w_in=(8.0, 0.5),
        w_ee=(0.1, 0.01),
        w_ei=(1.0, 0.1),
        w_ie=(0.3, 0.03),
        w_ii=(0.1, 0.01),
        trainable_w_ee=True,
        trainable_w_ei=True,
        trainable_w_ie=True,
        trainable_w_ii=True,
        readout_mode="mem-mean",
        refractory_e_ms=1.2,
        refractory_i_ms=0.6,
        refractory_policy="exact",
    )
    eager, compiled = build_net("ping", **kwargs), build_net("ping", **kwargs)
    compiled.load_state_dict(eager.state_dict())
    # CUDA would otherwise automatically compile the supposed eager reference.
    eager._compiled_cache["step"] = eager._step_body
    # CPU production normally skips compilation because some hosts cannot build
    # C++ kernels. Install actual Inductor here to test the same public loop and
    # step body used by compiled production, with no graph breaks permitted.
    compiled._compiled_cache["step"] = torch.compile(
        compiled._step_body, backend="inductor", fullgraph=True, dynamic=False
    )
    generator = torch.Generator().manual_seed(91)
    inputs = (
        (torch.rand((M.T_steps, 2, 3), generator=generator) < 0.4)
        .float()
        .to(device_name)
    )
    outcomes = []
    for net in (eager, compiled):
        net.recording = True
        output = net(input_spikes=inputs)
        output.square().mean().backward()
        gradients = {}
        for name, parameter in net.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            gradients[name] = parameter.grad.detach().clone()
        outcomes.append((output.detach(), net.spike_record, gradients))
    eager_output, eager_record, eager_grad = outcomes[0]
    compiled_output, compiled_record, compiled_grad = outcomes[1]
    assert eager_record.keys() == compiled_record.keys()
    for key in eager_record:
        if key in {"hid", "inh", "input", "out_spikes"}:
            assert torch.equal(eager_record[key], compiled_record[key]), key
        else:
            torch.testing.assert_close(
                compiled_record[key],
                eager_record[key],
                rtol=1e-5,
                atol=1e-5,
                msg=lambda message: f"{key}: {message}",
            )
    torch.testing.assert_close(compiled_output, eager_output, rtol=1e-5, atol=1e-5)
    assert eager_grad.keys() == compiled_grad.keys()
    assert len(eager_grad) == 6
    for key in eager_grad:
        torch.testing.assert_close(
            compiled_grad[key],
            eager_grad[key],
            rtol=1e-4,
            atol=1e-5,
            msg=lambda message: f"{key}: {message}",
        )
    for key, ref_ms in (("hid", 1.2), ("inh", 0.6)):
        assert compiled_record[key].sum() > 0, key
        for neuron in compiled_record[key].flatten(1).T:
            times = torch.where(neuron != 0)[0]
            assert torch.all(times.diff() >= round(ref_ms / dt))
