"""Per-model forward-pass smoke: shape + finiteness on tiny input.

Catches NaN blow-ups and shape regressions that the training e2e can mask
by retrying with different data.
"""

import pytest
import torch

from snnlab.sim import models as M
from snnlab.sim.config import build_net


@pytest.fixture(autouse=True)
def _tiny_model_sizes():
    old = (M.N_IN, M.N_HID, M.N_INH, M.N_OUT, M.HIDDEN_SIZES, M.T_ms, M.T_steps)
    M.N_IN = 8
    M.N_HID = 16
    M.N_INH = 4
    M.N_OUT = 10
    M.HIDDEN_SIZES = [16]
    M.T_ms = 20.0
    M.T_steps = int(M.T_ms / M.dt)
    yield
    (M.N_IN, M.N_HID, M.N_INH, M.N_OUT, M.HIDDEN_SIZES, M.T_ms, M.T_steps) = old


@pytest.mark.parametrize("model_name", ["ping"])
def test_forward_returns_finite_logits(model_name):
    net = build_net(model_name, hidden_sizes=[16])
    B = 2
    # Sparse random spike input, shape (T, B, N_IN)
    torch.manual_seed(0)
    spikes = (torch.rand(M.T_steps, B, M.N_IN) < 0.1).float()

    with torch.no_grad():
        out = net(input_spikes=spikes)

    assert out.shape == (B, M.N_OUT), f"{model_name} got {out.shape}"
    assert torch.isfinite(out).all(), f"{model_name} produced non-finite logits"


@pytest.mark.parametrize("model_name", ["ping"])
def test_forward_zero_input_is_finite(model_name):
    """Zero input shouldn't blow up (no NaNs from div-by-zero, etc.)."""
    net = build_net(model_name, hidden_sizes=[16])
    B = 2
    spikes = torch.zeros(M.T_steps, B, M.N_IN)
    with torch.no_grad():
        out = net(input_spikes=spikes)
    assert torch.isfinite(out).all()


def test_trainable_leak_and_adaptive_threshold_register_bounded_params():
    net = build_net(
        "ping",
        hidden_sizes=[16],
        train_leak=True,
        adaptive_threshold=True,
        adapt_strength_init_mv=1.5,
    )

    assert dict(net.named_parameters())["tau_m_e_logit.1"].shape == (16,)
    assert dict(net.named_parameters())["tau_m_i_logit.1"].shape == (4,)
    assert dict(net.named_parameters())["adapt_tau_logit.1"].shape == (16,)
    assert dict(net.named_parameters())["adapt_strength_logit.1"].shape == (16,)

    _c_e, g_l_e, _c_i, g_l_i = net.leak_params("1")
    tau_e = M.C_m_E / g_l_e
    tau_i = M.C_m_I / g_l_i
    assert torch.all(tau_e >= M.TRAINABLE_TAU_M_E_BOUNDS_MS[0])
    assert torch.all(tau_e <= M.TRAINABLE_TAU_M_E_BOUNDS_MS[1])
    assert torch.all(tau_i >= M.TRAINABLE_TAU_M_I_BOUNDS_MS[0])
    assert torch.all(tau_i <= M.TRAINABLE_TAU_M_I_BOUNDS_MS[1])

    decay, strength = net.adapt_params("1")
    assert torch.all(decay > 0)
    assert torch.all(decay < 1)
    assert torch.all(strength >= 0)
    assert torch.all(strength <= M.ADAPT_STRENGTH_MAX_MV)


@pytest.mark.parametrize("ei_strength", [0.0, 1.0])
def test_trainable_leak_and_adaptive_threshold_forward_backward(ei_strength):
    net = build_net(
        "ping",
        w_in=(10.0, 1.0),
        w_in_initial_zero_fraction=0.0,
        ei_strength=ei_strength,
        hidden_sizes=[16],
        train_leak=True,
        adaptive_threshold=True,
        readout_mode="cumulative-potential",
        signed_readout=True,
        readout_bias=True,
    )
    B = 2
    torch.manual_seed(1)
    spikes = (torch.rand(M.T_steps, B, M.N_IN) < 0.4).float()

    out = net(input_spikes=spikes)
    loss = out.square().mean()
    loss.backward()

    assert out.shape == (B, M.N_OUT)
    assert torch.isfinite(out).all()
    assert torch.isfinite(loss)
    for name, parameter in net.named_parameters():
        if (
            name.startswith("tau_m_")
            or name.startswith("adapt_tau_")
            or name.startswith("adapt_strength_")
        ):
            assert parameter.grad is not None, f"{name} did not receive a gradient"
            assert torch.isfinite(parameter.grad).all(), name
