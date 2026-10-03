"""Physical refractory durations must reach the public network update path."""

import math

import pytest
import torch

from snnlab.sim import models as M
from snnlab.sim.config import build_net, set_sim_dt


@pytest.mark.parametrize("integrator", ["expeuler", "fwd"])
@pytest.mark.parametrize("dt", [0.05, 0.1, 0.2, 0.3, 0.6])
def test_public_forward_enforces_physical_refractory_counts(
    monkeypatch, integrator, dt
):
    monkeypatch.setattr(M, "N_IN", 1)
    monkeypatch.setattr(M, "N_OUT", 1)
    monkeypatch.setattr(M, "COBA_INTEGRATOR", integrator)
    # Stale globals must not influence either population's production update.
    monkeypatch.setattr(M, "ref_steps_E", 999)
    monkeypatch.setattr(M, "ref_steps_I", 999)
    set_sim_dt(dt, 6.0)
    net = build_net(
        "ping",
        hidden_sizes=[4],
        w_ee=(0, 0),
        w_ei=(0, 0),
        w_ie=(0, 0),
        w_ii=(0, 0),
        refractory_e_ms=1.2,
        refractory_i_ms=0.6,
        refractory_policy="exact",
    )
    net.recording = True
    with torch.no_grad():
        net(
            ext_g=torch.full((M.T_steps, 4), 100.0),
            ext_g_i=torch.full((M.T_steps, 1), 100.0),
        )
    for key, expected_steps in (("hid", round(1.2 / dt)), ("inh", round(0.6 / dt))):
        times = torch.where(net.spike_record[key][:, 0] != 0)[0]
        assert len(times) >= 5
        assert times[0] == 0
        assert torch.all(times.diff() == expected_steps)
    assert net.timing_metadata["refractory_e_steps"] == round(1.2 / dt)
    assert net.timing_metadata["refractory_i_steps"] == round(0.6 / dt)


@pytest.mark.parametrize("integrator", ["expeuler", "fwd"])
@pytest.mark.parametrize("dt", [0.05, 0.1, 0.2, 0.3, 0.6])
def test_public_forward_holds_reset_then_releases_voltage(monkeypatch, integrator, dt):
    """After one forced spike, subthreshold drive reveals the first free update."""
    monkeypatch.setattr(M, "N_IN", 1)
    monkeypatch.setattr(M, "N_OUT", 1)
    monkeypatch.setattr(M, "COBA_INTEGRATOR", integrator)
    set_sim_dt(dt, 2.4)
    net = build_net(
        "ping",
        hidden_sizes=[4],
        w_ee=(0, 0),
        w_ei=(0, 0),
        w_ie=(0, 0),
        w_ii=(0, 0),
        refractory_e_ms=1.2,
        refractory_i_ms=0.6,
        refractory_policy="exact",
    )
    net.recording = True
    decay = math.exp(-dt / M.tau_ampa)
    # External inputs add conductance to its decaying state. Cancel the first
    # pulse's tail so a persistent subthreshold conductance exposes release.
    kicks = torch.full((M.T_steps,), 0.01 * (1.0 - decay))
    kicks[0] = 100.0
    kicks[1] = 0.01 - 100.0 * decay
    with torch.no_grad():
        net(
            ext_g=kicks[:, None].expand(-1, 4),
            ext_g_i=kicks[:, None],
        )
    for spike_key, voltage_key, ref_ms in (
        ("hid", "v_e_1", 1.2),
        ("inh", "v_i_1", 0.6),
    ):
        count = round(ref_ms / dt)
        spikes = net.spike_record[spike_key][:, 0]
        voltage = net.spike_record[voltage_key][:, 0]
        assert torch.equal(torch.where(spikes != 0)[0], torch.tensor([0]))
        assert torch.all(voltage[:count] == M.V_reset)
        assert M.V_reset < voltage[count] < M.V_th
        assert torch.all(voltage[count:] > M.V_reset)


def test_exact_model_rejects_unrepresentable_runtime_timestep(monkeypatch):
    monkeypatch.setattr(M, "N_IN", 1)
    monkeypatch.setattr(M, "N_OUT", 1)
    set_sim_dt(0.25, 1.0)
    net = build_net(
        "ping",
        hidden_sizes=[4],
        refractory_e_ms=1.2,
        refractory_i_ms=0.6,
        refractory_policy="exact",
    )
    with pytest.raises(ValueError, match="not a positive whole number"):
        net(ext_g=torch.zeros(M.T_steps, 4))


def test_generic_defaults_and_model_specific_values_do_not_share_state():
    generic = build_net("ping", hidden_sizes=[4])
    collection = build_net(
        "ping",
        hidden_sizes=[4],
        refractory_e_ms=1.2,
        refractory_i_ms=0.6,
        refractory_policy="exact",
    )
    assert (generic.refractory_e_ms, generic.refractory_i_ms) == (3.0, 1.5)
    assert generic.refractory_policy == "nearest"
    assert (collection.refractory_e_ms, collection.refractory_i_ms) == (1.2, 0.6)


def test_rates_use_realized_trial_duration(monkeypatch):
    monkeypatch.setattr(M, "N_IN", 1)
    monkeypatch.setattr(M, "N_OUT", 1)
    set_sim_dt(0.6, 2.0)
    net = build_net(
        "ping",
        hidden_sizes=[4],
        refractory_e_ms=1.2,
        refractory_i_ms=0.6,
        refractory_policy="exact",
    )
    net.recording = True
    with torch.no_grad():
        net(ext_g=torch.full((M.T_steps, 4), 100.0))
    expected = net.spike_record["hid"].sum().item() / (4 * 1.8 / 1000)
    assert net.rates["hid"] == pytest.approx(expected)
    assert net.timing_metadata["nominal_duration_ms"] == 2.0
    assert net.timing_metadata["duration_steps"] == 3
    assert net.timing_metadata["realized_duration_ms"] == pytest.approx(1.8)
