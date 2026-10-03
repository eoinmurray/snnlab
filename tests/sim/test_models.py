import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from snnlab.sim import models as M
from snnlab.sim.config import build_net
from snnlab.sim.infer import probe


def test_external_inhibitory_conductance_enters_gaba_state_without_reset():
    M.N_IN = 4
    M.N_OUT = 2
    M.T_steps = 4
    M.dt = 0.25
    net = M.COBANet(hidden_sizes=[4], w_in=(0, 0), w_hid=(0, 0))
    net.recording = True
    zeros = torch.zeros(4, 4)
    inhibitory_e = torch.full((4, 4), 0.05)
    inhibitory_i = torch.full((4, 1), 0.03)
    net(input_spikes=zeros, ext_g_inhib_e=inhibitory_e, ext_g_inhib_i=inhibitory_i)
    rec = net.spike_record
    assert torch.all(rec["gi_e_1"][1:] > 0)
    assert torch.all(rec["gi_i_1"][1:] > 0)
    assert torch.all(rec["ge_e_1"] == 0)


@pytest.fixture(autouse=True)
def _small_model_sizes():
    """Keep tests fast by shrinking model constants."""
    old = (M.N_IN, M.N_HID, M.N_INH, M.N_OUT, M.HIDDEN_SIZES, M.T_ms, M.T_steps)
    M.N_IN = 16
    M.N_HID = 32
    M.N_INH = 8
    M.N_OUT = 10
    M.HIDDEN_SIZES = [32]
    M.T_ms = 50.0
    M.T_steps = int(M.T_ms / M.dt)
    yield
    (M.N_IN, M.N_HID, M.N_INH, M.N_OUT, M.HIDDEN_SIZES, M.T_ms, M.T_steps) = old


class TestBuildNetRegistry:
    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            build_net("nonexistent")

    def test_ping_instantiates(self):
        net = build_net("ping", hidden_sizes=[32])
        assert isinstance(net, nn.Module)


class TestCOBANetFrozenWeights:
    def test_recurrent_weights_are_frozen(self):
        """COBANet W_ee/W_ei/W_ie must have requires_grad=False after construction."""
        net = build_net("ping", hidden_sizes=[32])
        for name in ["W_ee", "W_ei", "W_ie"]:
            pdict = getattr(net, name)
            assert isinstance(pdict, nn.ParameterDict)
            for key, param in pdict.items():
                assert not param.requires_grad, (
                    f"{name}[{key}] should be frozen (requires_grad=False)"
                )

    def test_recurrent_weights_survive_optimizer_step(self):
        """Freezing means an optimizer.step() on all params leaves them unchanged."""
        net = build_net("ping", hidden_sizes=[32])
        snapshots = {
            f"{name}_{k}": p.detach().clone()
            for name in ["W_ee", "W_ei", "W_ie"]
            for k, p in getattr(net, name).items()
        }
        trainable = [p for p in net.parameters() if p.requires_grad]
        assert len(trainable) > 0, "expected at least one trainable param"
        opt = torch.optim.SGD(trainable, lr=1.0)
        for p in trainable:
            p.grad = torch.ones_like(p)
        opt.step()
        for name in ["W_ee", "W_ei", "W_ie"]:
            for k, p in getattr(net, name).items():
                assert torch.equal(p, snapshots[f"{name}_{k}"]), (
                    f"{name}[{k}] changed after optimizer step"
                )


class TestSeedReproducibility:
    def test_same_seed_gives_same_weights(self):
        def _weights(net):
            return [p.detach().clone() for p in net.parameters()]

        torch.manual_seed(123)
        a = _weights(build_net("ping", hidden_sizes=[32]))
        torch.manual_seed(123)
        b = _weights(build_net("ping", hidden_sizes=[32]))
        assert len(a) == len(b)
        for pa, pb in zip(a, b):
            assert torch.equal(pa, pb)


class TestReadoutInitialization:
    def test_direct_stored_initializer_matches_accepted_legacy_recipe(self):
        shape = (1024, 10)
        torch.manual_seed(42)
        legacy = M.init_weight(shape, "normal", 5.1, 3.8) * 225
        torch.manual_seed(42)
        direct = M.init_readout_weight(
            shape,
            mean=5.1 * 225 / 1024,
            std=3.8 * 225 / 1024,
        )
        torch.testing.assert_close(direct, legacy)

    def test_direct_initializer_preserves_full_model_seed_equivalence(self):
        torch.manual_seed(42)
        legacy = build_net("ping", hidden_sizes=[32])
        legacy_readout = legacy.W_ff[-1].detach() * 225
        torch.manual_seed(42)
        direct = build_net(
            "ping",
            hidden_sizes=[32],
            readout_w_init=(5.1 * 225 / 32, 3.8 * 225 / 32),
        )
        torch.testing.assert_close(direct.W_ff[-1], legacy_readout)

    def test_direct_initializer_is_applied_to_stored_readout_only(self):
        torch.manual_seed(42)
        net = build_net(
            "ping",
            hidden_sizes=[32],
            readout_w_init=(1.25, 0.0),
        )
        assert torch.all(net.W_ff[-1] == 1.25)
        assert not torch.all(net.W_ff[0] == 1.25)

    def test_direct_initializer_rejects_signed_readout(self):
        with pytest.raises(ValueError, match="cannot be combined"):
            build_net(
                "ping",
                hidden_sizes=[32],
                signed_readout=True,
                readout_w_init=(1.25, 0.1),
            )


class TestWeightInitializationContract:
    @pytest.mark.parametrize("fraction", [-0.1, 1.0, 1.1])
    def test_initial_zero_fraction_rejects_invalid_values(self, fraction):
        with pytest.raises(ValueError, match="0 <= fraction < 1"):
            M.init_weight((4, 2), initial_zero_fraction=fraction)

    @pytest.mark.parametrize("exact_k", [False, True])
    def test_renamed_initializer_is_bit_identical_to_previous_recipe(self, exact_k):
        shape = (64, 8)
        fraction = 0.75

        def previous_recipe():
            n_pre, n_post = shape
            weight = torch.randn(*shape).mul_(0.2).add_(0.9).clamp_(min=0)
            if exact_k:
                k = max(1, int(round((1.0 - fraction) * n_pre)))
                mask = torch.zeros(n_pre, n_post)
                for column in range(n_post):
                    indices = torch.randperm(n_pre)[:k]
                    mask[indices, column] = 1.0
                weight = weight * mask * (n_pre / k)
            else:
                weight = weight * (torch.rand(*shape) > fraction).float()
                weight = weight / (1.0 - fraction)
            return weight / n_pre

        try:
            M.EXACT_K_INITIALIZATION = exact_k
            torch.manual_seed(29)
            expected = previous_recipe()
            torch.manual_seed(29)
            actual = M.init_weight(shape, "lower_clamped_normal", 0.9, 0.2, fraction)
        finally:
            M.EXACT_K_INITIALIZATION = False
        assert torch.equal(actual, expected)

    def test_lower_clamped_normal_and_initial_zeroing_are_separate(self):
        M.EXACT_K_INITIALIZATION = False
        torch.manual_seed(7)
        weight, provenance = M.init_weight(
            (20_000, 2),
            "lower_clamped_normal",
            0.0,
            1.0,
            0.25,
            return_provenance=True,
        )
        stats = provenance["statistics"]
        assert provenance["zeros_remain_trainable"] is True
        assert provenance["distribution"] == "lower_clamped_normal"
        assert stats["explicit_zero_fraction"] == pytest.approx(0.25, abs=0.01)
        assert stats["lower_clamp_zero_fraction_of_unzeroed"] == pytest.approx(
            0.5, abs=0.01
        )
        assert stats["initialization_zero_fraction"] > 0.5
        assert weight.shape == (20_000, 2)

    @pytest.mark.parametrize("zero_source", ["lower_clamp", "explicit_zeroing"])
    def test_every_initialization_zero_can_regrow(self, zero_source):
        torch.manual_seed(11)
        if zero_source == "lower_clamp":
            weight = M.init_weight((64, 4), "lower_clamped_normal", 0.0, 1.0)
        else:
            weight = M.init_weight((64, 4), "lower_clamped_normal", 5.0, 0.0, 0.5)
        parameter = nn.Parameter(weight)
        initial_zeros = parameter.detach() == 0
        assert initial_zeros.any()
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        parameter.grad = torch.full_like(parameter, -1.0)
        optimizer.step()
        assert torch.all(parameter.detach()[initial_zeros] > 0)

    def test_provenance_matches_tensor_and_expected_summed_coupling(self):
        torch.manual_seed(3)
        weight, provenance = M.init_weight(
            (10_000, 3),
            "lower_clamped_normal",
            0.9,
            0.09,
            0.95,
            return_provenance=True,
        )
        stats = provenance["statistics"]
        assert stats["all_entries"]["mean"] == pytest.approx(float(weight.mean()))
        assert stats["initialization_zero_count"] == int((weight == 0).sum())
        assert stats["realized_column_sum"]["mean"] == pytest.approx(
            provenance["expected_summed_coupling_after_clamp"], rel=0.03
        )


class TestFeedforwardDalesClamp:
    """The forward clamp applies specifically to feedforward ``W_ff``.

    Recurrent conductances are used directly and kept non-negative by the
    post-optimiser ``project_dales()`` call.
    """

    def test_negative_w_ff_zeroes_out_in_forward(self):
        """If Dale's law is on, replacing a positive W_ff entry with a
        large negative one shouldn't change forward output — the clamp
        treats it as zero."""
        torch.manual_seed(0)
        net = build_net("ping", hidden_sizes=[32], dales_law=True)
        net.recording = False
        assert not net.signed_weights, (
            "dales_law=True should produce signed_weights=False"
        )
        spikes = (torch.rand(M.T_steps, 1, M.N_IN) < 0.2).float()
        with torch.no_grad():
            ref_logits = net.forward(input_spikes=spikes)

        # Zero a few W_ff[0] entries in storage, run again — should match.
        with torch.no_grad():
            net.W_ff[0].data[:3, :3] = 0.0
            forced_zero_logits = net.forward(input_spikes=spikes)

        # Now overwrite those same entries with large negatives and rerun
        # — the clamp on forward must collapse them to zero, producing
        # the same output as the explicit-zero case.
        with torch.no_grad():
            net.W_ff[0].data[:3, :3] = -100.0
            clamped_logits = net.forward(input_spikes=spikes)

        assert torch.allclose(forced_zero_logits, clamped_logits, atol=1e-6), (
            "negative W_ff entries should clamp to zero in forward, but the "
            "output differed from explicit-zero entries"
        )
        # And the ref (with the original positive values) should differ from
        # both — otherwise the test is vacuous (the W entries we zeroed had
        # no effect on the output).
        assert not torch.allclose(ref_logits, clamped_logits, atol=1e-6), (
            "test is vacuous: zeroing some W_ff entries had no effect on output"
        )

    def test_signed_weights_disables_clamp(self):
        """With dales_law=False, the forward pass uses raw signed weights.
        Negative entries are NOT zeroed out, so they must propagate."""
        torch.manual_seed(0)
        net = build_net("ping", hidden_sizes=[32], dales_law=False)
        assert net.signed_weights
        spikes = (torch.rand(M.T_steps, 1, M.N_IN) < 0.2).float()

        with torch.no_grad():
            net.W_ff[0].data[:3, :3] = 0.0
            zero_logits = net.forward(input_spikes=spikes)
            net.W_ff[0].data[:3, :3] = -100.0
            neg_logits = net.forward(input_spikes=spikes)

        # The two should differ — without the clamp, -100 ≠ 0.
        assert not torch.allclose(zero_logits, neg_logits, atol=1e-6), (
            "signed_weights=True should let negative W_ff entries change "
            "the forward output, but zero and -100 gave identical logits"
        )


class TestCumulativePotentialReadout:
    def _net(self, **kwargs):
        return build_net(
            "ping",
            hidden_sizes=[32],
            readout_mode="cumulative-potential",
            signed_readout=True,
            readout_bias=True,
            **kwargs,
        )

    def test_matches_reference_recurrence_timestep_by_timestep(self):
        """Recorded scores equal sum_t softmax(leaky potential_t)."""
        M.T_steps = 4
        M.T_ms = M.T_steps * M.dt
        torch.manual_seed(7)
        net = self._net()
        net.recording = True
        spikes = (torch.rand(M.T_steps, 2, M.N_IN) < 0.5).float()

        result = net(input_spikes=spikes)
        hidden = net.spike_record[net._hid_key(1)]
        alpha_lo = torch.exp(torch.tensor(-M.dt / 5.0))
        alpha_hi = torch.exp(torch.tensor(-M.dt / 25.0))
        alpha = net.readout_alpha.clamp(min=alpha_lo, max=alpha_hi)
        potential = torch.zeros(2, M.N_OUT)
        evidence = torch.zeros(2, M.N_OUT)
        expected = []
        for t in range(M.T_steps):
            drive = hidden[t] @ net.W_ff[-1] + net.b_out
            potential = alpha * potential + (1.0 - alpha) * drive
            evidence = evidence + F.softmax(potential, dim=1)
            expected.append(evidence)
        expected = torch.stack(expected)

        assert torch.allclose(net.spike_record["out"], expected, atol=1e-6)
        assert torch.allclose(result, expected[-1], atol=1e-6)
        expected_mass = torch.arange(1, M.T_steps + 1, dtype=torch.float32)
        expected_mass = expected_mass.unsqueeze(1).expand(-1, 2)
        assert torch.allclose(expected.sum(dim=2), expected_mass, atol=1e-6)

    def test_signed_decoder_is_exempt_from_dale_projection(self):
        net = self._net()
        with torch.no_grad():
            net.W_ff[0].fill_(-1.0)
            net.W_ff[-1].fill_(-1.0)
            net.b_out.fill_(-1.0)
        net.project_dales()

        assert torch.all(net.W_ff[0] == 0)
        assert torch.all(net.W_ff[-1] == -1)
        assert torch.all(net.b_out == -1)

    def test_decoder_parameters_receive_gradients(self):
        M.T_steps = 4
        M.T_ms = M.T_steps * M.dt
        torch.manual_seed(11)
        net = self._net()
        logits = net(input_spikes=torch.ones(M.T_steps, 2, M.N_IN))
        F.cross_entropy(logits, torch.tensor([0, 1])).backward()

        for name, parameter in (
            ("decoder weights", net.W_ff[-1]),
            ("decoder bias", net.b_out),
            ("decoder alpha", net.readout_alpha),
        ):
            assert parameter.grad is not None, f"missing {name} gradient"
            assert torch.isfinite(parameter.grad).all(), f"non-finite {name} gradient"

    def test_bias_rejected_for_legacy_readouts(self):
        with pytest.raises(ValueError, match="only by the cumulative-potential"):
            build_net("ping", hidden_sizes=[32], readout_mode="rate", readout_bias=True)


class TestSpikeRateReadout:
    def _always_driven_net(self):
        net = build_net("ping", hidden_sizes=[32], readout_mode="spike-rate")
        with torch.no_grad():
            net.W_ff[-1].fill_(100.0)

        def force_hidden_spikes(s_e, s_i, _layer):
            return torch.ones_like(s_e), s_i

        net._hidden_perturb_fn = force_hidden_spikes
        return net

    def test_logits_are_recorded_output_spikes_per_second(self):
        M.T_steps = 8
        M.T_ms = M.T_steps * M.dt
        net = self._always_driven_net()
        net.recording = True

        logits = net(input_spikes=torch.zeros(M.T_steps, 2, M.N_IN))
        recorded = net.spike_record["out_spikes"]
        expected = recorded.sum(dim=0) / (M.T_steps * M.dt / 1000.0)

        assert torch.equal(recorded, recorded.bool().float())
        assert torch.allclose(logits, expected)
        assert net.spike_record["v_out"].shape == recorded.shape
        expected_population_rate = recorded.detach().mean().item() * 1000.0 / M.dt
        assert net.rates["out"] == pytest.approx(expected_population_rate)

    def test_duration_normalization_keeps_constant_output_rate_constant(self):
        rates = []
        for steps in (4, 12):
            M.T_steps = steps
            M.T_ms = steps * M.dt
            net = self._always_driven_net()
            rates.append(net(input_spikes=torch.zeros(steps, 1, M.N_IN)))

        assert torch.allclose(rates[0], rates[1])


class TestSpikeCountReadout:
    def test_logits_are_recorded_output_spike_counts(self):
        M.T_steps = 8
        M.T_ms = M.T_steps * M.dt
        net = build_net("ping", hidden_sizes=[32], readout_mode="spike-count")
        with torch.no_grad():
            net.W_ff[-1].fill_(100.0)

        def force_hidden_spikes(s_e, s_i, _layer):
            return torch.ones_like(s_e), s_i

        net._hidden_perturb_fn = force_hidden_spikes
        net.recording = True
        logits = net(input_spikes=torch.zeros(M.T_steps, 2, M.N_IN))
        recorded = net.spike_record["out_spikes"]

        assert torch.equal(logits, recorded.sum(dim=0))
        assert torch.equal(net.last_output_spike_counts, logits)
        assert net.spike_record["v_out"].shape == recorded.shape

    def test_surrogate_spikes_propagate_gradient_to_output_weights(self):
        M.T_steps = 6
        M.T_ms = M.T_steps * M.dt
        net = build_net("ping", hidden_sizes=[32], readout_mode="spike-rate")
        logits = net(input_spikes=torch.ones(M.T_steps, 2, M.N_IN))
        F.cross_entropy(logits, torch.tensor([0, 1])).backward()

        assert net.W_ff[-1].grad is not None
        assert torch.isfinite(net.W_ff[-1].grad).all()

    def test_boundary_reset_restarts_only_output_lif(self):
        M.T_steps = 8
        M.T_ms = M.T_steps * M.dt
        torch.manual_seed(19)
        baseline = build_net("ping", hidden_sizes=[32], readout_mode="spike-rate")
        reset_net = build_net("ping", hidden_sizes=[32], readout_mode="spike-rate")
        reset_net.load_state_dict(baseline.state_dict())
        baseline.recording = reset_net.recording = True
        spikes = torch.ones(M.T_steps, 1, M.N_IN)

        baseline(input_spikes=spikes)
        reset_mask = torch.zeros(M.T_steps, dtype=torch.bool)
        reset_mask[4] = True
        reset_net(input_spikes=spikes, readout_reset_mask=reset_mask)

        assert torch.equal(baseline.spike_record["hid"], reset_net.spike_record["hid"])
        assert torch.equal(baseline.spike_record["inh"], reset_net.spike_record["inh"])
        assert torch.all(reset_net.spike_record["v_out"][4] <= 1.0)

    def test_input_file_probe_restores_mode_and_emits_output_raster(self, tmp_path):
        M.N_IN = 8
        M.T_steps = 6
        M.T_ms = M.T_steps * M.dt
        net = build_net("ping", hidden_sizes=[16], readout_mode="spike-rate")
        weights = tmp_path / "weights.pth"
        torch.save(net.state_dict(), weights)
        input_file = tmp_path / "input.npz"
        np.savez(
            input_file,
            input_spikes=np.ones((6, 1, 8), dtype=np.float32),
            readout_reset=np.asarray([True, False, False, True, False, False]),
        )

        probe(
            model_name="ping",
            dt=M.dt,
            t_ms=M.T_ms,
            hidden_sizes=[16],
            n_in=8,
            load_weights=weights,
            input_file=input_file,
            out_dir=tmp_path,
            outputs={"rasters"},
            readout_mode="spike-rate",
        )

        raster = np.load(tmp_path / "rasters.npz")
        assert {"out_trial", "out_t", "out_cell"} <= set(raster.files)
        assert int(raster["T"]) == 6

    def test_input_file_probe_emits_compact_batched_spike_summary(self, tmp_path):
        M.N_IN = 8
        M.T_steps = 6
        M.T_ms = M.T_steps * M.dt
        net = build_net("ping", hidden_sizes=[16], readout_mode="spike-count")
        weights = tmp_path / "weights.pth"
        torch.save(net.state_dict(), weights)
        input_file = tmp_path / "input.npz"
        np.savez(
            input_file,
            input_spikes=np.ones((6, 3, 8), dtype=np.float32),
            readout_reset=np.asarray([True, False, False, True, False, False]),
        )

        probe(
            model_name="ping",
            dt=M.dt,
            t_ms=M.T_ms,
            hidden_sizes=[16],
            n_in=8,
            load_weights=weights,
            input_file=input_file,
            out_dir=tmp_path,
            outputs={"spike_summary", "rasters"},
            readout_mode="spike-count",
        )

        summary = np.load(tmp_path / "spike_summary.npz")
        assert summary["e_counts"].shape == (3, 2)
        assert summary["i_counts"].shape == (3, 2)
        assert summary["out_counts"].shape == (3, 2, M.N_OUT)
        assert summary["segment_starts"].tolist() == [0, 3]
        assert summary["segment_stops"].tolist() == [3, 6]
        raster = np.load(tmp_path / "rasters.npz")
        expected = np.zeros((3, 2, M.N_OUT), dtype=np.int64)
        for trial, timestep, cell in zip(
            raster["out_trial"], raster["out_t"], raster["out_cell"], strict=True
        ):
            expected[int(trial), int(timestep) // 3, int(cell)] += 1
        np.testing.assert_array_equal(summary["out_counts"], expected)

    def test_batched_streams_match_independent_single_streams(self):
        M.N_IN = 8
        M.T_steps = 12
        M.T_ms = M.T_steps * M.dt
        torch.manual_seed(82)
        batched = build_net("ping", hidden_sizes=[16], readout_mode="spike-count")
        batched.recording = True
        inputs = (torch.rand(M.T_steps, 3, M.N_IN) < 0.25).float()
        reset = torch.zeros(M.T_steps, dtype=torch.bool)
        reset[[0, 4, 8]] = True
        batched(input_spikes=inputs, readout_reset_mask=reset)

        expected = {name: [] for name in ("hid", "inh", "out_spikes")}
        for trial in range(inputs.shape[1]):
            single = build_net("ping", hidden_sizes=[16], readout_mode="spike-count")
            single.load_state_dict(batched.state_dict())
            single.recording = True
            single(
                input_spikes=inputs[:, trial : trial + 1],
                readout_reset_mask=reset,
            )
            for name in expected:
                expected[name].append(single.spike_record[name])
        for name, trials in expected.items():
            torch.testing.assert_close(
                batched.spike_record[name], torch.stack(trials, dim=1), rtol=0, atol=0
            )


class TestRecurrentDalesProjection:
    def _trainable_net(self, *, dales_law=True):
        return build_net(
            "ping",
            hidden_sizes=[32],
            dales_law=dales_law,
            trainable_w_ee=True,
            trainable_w_ei=True,
            trainable_w_ie=True,
            trainable_w_ii=True,
        )

    def test_project_dales_projects_every_constrained_matrix(self):
        net = self._trainable_net()
        constrained = list(net.W_ff) + [
            p
            for name in ("W_ee", "W_ei", "W_ie", "W_ii")
            for p in getattr(net, name).values()
        ]
        with torch.no_grad():
            for p in constrained:
                p.fill_(-1.0)

        net.project_dales()

        assert all(torch.count_nonzero(p).item() == 0 for p in constrained)
        assert all(torch.all(p >= 0) for p in net.W_ie.values())

    def test_project_dales_leaves_frozen_recurrence_untouched(self):
        net = build_net("ping", hidden_sizes=[32], dales_law=True)
        with torch.no_grad():
            net.W_ie["1"].fill_(-1.0)

        net.project_dales()

        assert torch.all(net.W_ie["1"] == -1.0)

    def test_signed_mode_is_not_projected(self):
        net = self._trainable_net(dales_law=False)
        with torch.no_grad():
            net.W_ie["1"].fill_(-1.0)

        net.project_dales()

        assert torch.all(net.W_ie["1"] == -1.0)

    def test_negative_optimizer_update_is_projected_to_zero(self):
        net = self._trainable_net()
        param = net.W_ie["1"]
        with torch.no_grad():
            param.fill_(0.25)
        opt = torch.optim.SGD([param], lr=1.0)
        opt.register_step_post_hook(lambda *_: net.project_dales())
        param.grad = torch.ones_like(param)

        opt.step()

        assert torch.count_nonzero(param).item() == 0

    def test_positive_w_ie_increases_gaba_conductance(self):
        spikes_i = torch.tensor([[1.0, 0.0]])
        w_ie = torch.tensor([[0.4, 0.2], [0.3, 0.1]])
        g_i_before = torch.zeros(1, 2)

        g_i_after = g_i_before + spikes_i @ w_ie

        assert torch.all(g_i_after > g_i_before)

    def test_positive_gaba_conductance_is_hyperpolarising_above_e_i(self):
        v = torch.tensor([M.E_i + 10.0])
        current = M.coba_current(torch.zeros_like(v), v, torch.tensor([0.5]))

        assert current.item() < 0
