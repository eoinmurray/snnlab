"""Compatibility gates for the additive snnlang bundle frontend."""

from __future__ import annotations

import json
import os
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from snnlab import lang as snn
from snnlab.sim import config
from snnlab.sim import models as M
from snnlab.sim.bundle import (
    BundleCompatibilityError,
    load_graph_bundle,
    load_simulation_recipe,
    load_training_recipe,
    translate_cobanet_v1,
    translate_training_v1,
)
from snnlab.sim.simulation_inputs import realize_simulation_inputs
from snnlab.sim.tool import _bundle_transition_schedule, parse_args
from tests._circuits import author_ping
from tests.sim._bundle_builders import ping_classifier


def _write_bundle(tmp_path):
    return ping_classifier().write(tmp_path / "network.bundle")


def _write_transition_bundle(tmp_path, name, w_ee):
    net = snn.Network(name, dt=0.25 * snn.ms)
    source = net.input(
        "input", shape=("time", "batch", 400), signal_type="spikes", unit="spike"
    )

    def sparse(mean, std):
        return snn.LowerClampedNormal(
            mean, std, initial_zero_fraction=0.975, zeroing="exact_k"
        )

    cell = author_ping(
        net,
        name="circuit",
        n_e=400,
        n_i=100,
        source=source,
        w_in=snn.Normal(0.01, 0.001),
        w_ee=sparse(*w_ee),
        w_ei=sparse(0.6, 0.18),
        w_ie=sparse(3.0, 0.9),
        w_ii=sparse(0.4, 0.12),
    )
    readout = snn.readouts.MeanVoltage(
        source=cell.E.spikes,
        classes=10,
        name="readout",
        tau=2 * snn.ms,
        weight=snn.Normal(5.1, 3.8),
    )
    net.output("logits", readout)
    return snn.compile(net).write(tmp_path / f"{name}.bundle")


def _legacy_argv(mode="sim"):
    return [
        mode,
        "--n-hidden",
        "256",
        "--readout",
        "mem-mean",
        "--dt",
        "0.1",
        "--refractory-e-ms",
        "1.2",
        "--refractory-i-ms",
        "0.6",
        "--refractory-policy",
        "exact",
        "--w-in",
        "0.2",
        "0.03",
        "--w-in-initial-zero-fraction",
        "0",
        "--ei-strength",
        "0.5",
        "--ei-ratio",
        "2",
        "--tau-gaba",
        "9",
    ]


def test_legacy_parse_defaults_are_unchanged():
    args = parse_args(["sim"])
    assert args.bundle is None
    assert args.model == "ping"
    assert args.n_hidden is None
    assert args.dt == pytest.approx(0.25)
    assert args.readout_mode == "rate"


@pytest.mark.parametrize("executor", ["legacy", "graph"])
@pytest.mark.parametrize(
    "override",
    [
        "--refractory-e-ms=1.2",
        "--refractory-i-ms=0.6",
        "--refractory-policy=exact",
    ],
)
def test_bundle_owns_refractory_settings(tmp_path, executor, override, capsys):
    root = _write_bundle(tmp_path)
    with pytest.raises(SystemExit):
        parse_args(["sim", "--bundle", str(root), "--executor", executor, override])
    assert "owns" in capsys.readouterr().err


def test_bundle_translates_to_same_structural_arguments_as_legacy(tmp_path):
    root = _write_bundle(tmp_path)
    bundle = parse_args(["sim", "--bundle", str(root)])
    legacy = parse_args(_legacy_argv())
    legacy.w_ei = [0.5, 0.05]
    legacy.w_ie = [1.0, 0.1]
    fields = (
        "model",
        "n_hidden",
        "readout_mode",
        "dt",
        "refractory_policy",
        "w_in",
        "w_in_initial_zero_fraction",
        "w_ei",
        "w_ie",
        "ei_strength",
        "ei_ratio",
        "recurrent_initial_zero_fraction",
        "tau_gaba",
    )
    assert {field: getattr(bundle, field) for field in fields} == {
        field: getattr(legacy, field) for field in fields
    }
    assert bundle.refractory_e_ms == pytest.approx(legacy.refractory_e_ms)
    assert bundle.refractory_i_ms == pytest.approx(legacy.refractory_i_ms)


def test_bundle_owns_four_recurrent_blocks_and_exact_k(tmp_path):
    net = snn.Network("four_block", dt=0.25 * snn.ms)
    source = net.input(
        "input", shape=("time", "batch", 400), signal_type="spikes", unit="spike"
    )

    def sparse(mean, std):
        return snn.LowerClampedNormal(
            mean, std, initial_zero_fraction=0.975, zeroing="exact_k"
        )

    cell = author_ping(
        net,
        name="circuit",
        n_e=400,
        n_i=100,
        source=source,
        w_in=snn.Normal(0.01, 0.001),
        w_ee=sparse(0.4, 0.12),
        w_ei=sparse(0.6, 0.18),
        w_ie=sparse(3.0, 0.9),
        w_ii=sparse(0.4, 0.12),
    )
    readout = snn.readouts.MeanVoltage(
        source=cell.E.spikes,
        classes=10,
        name="readout",
        tau=2 * snn.ms,
        weight=snn.Normal(5.1, 3.8),
    )
    net.output("logits", readout)
    root = snn.compile(net).write(tmp_path / "four.bundle")

    args = parse_args(["sim", "--bundle", str(root)])
    assert args.n_in == 400
    assert args.n_hidden == [400]
    assert args.w_ee == [0.4, 0.12]
    assert args.w_ei == [0.6, 0.18]
    assert args.w_ie == [3.0, 0.9]
    assert args.w_ii == [0.4, 0.12]
    assert args.recurrent_initial_zero_fraction == pytest.approx(0.975)
    assert args.exact_k_initialization is True


def _write_combined_bundle(tmp_path):
    net = snn.Network("combined", dt=0.25 * snn.ms)
    source = net.input(
        "input", shape=("time", "batch", 400), signal_type="spikes", unit="spike"
    )
    cell = author_ping(
        net,
        name="circuit",
        n_e=400,
        n_i=100,
        source_e=source,
        source_i=source,
        w_in_e=snn.Normal(0.01, 0.001),
        w_in_i=snn.Normal(0.02, 0.002),
    )
    readout = snn.readouts.MeanVoltage(
        source=cell.E.spikes,
        classes=10,
        name="readout",
        tau=2 * snn.ms,
        weight=snn.Normal(5.1, 3.8),
    )
    net.output("logits", readout)

    def channel(tau):
        return snn.BackgroundChannel(
            private=snn.ShotNoise(200, 0.01, tau),
            shared=snn.GlobalShotNoise(80, 0.02, tau),
            heterogeneity=snn.CellDistribution(
                rate=snn.Uniform(0.8, 1.2), amplitude=snn.Uniform(0.9, 1.1)
            ),
        )

    simulation = snn.SimulationSpec(
        spike_sources=[snn.StructuredPoisson(source, 10)],
        backgrounds=[
            snn.ConductanceBackground(cell.E, channel(2), channel(9)),
            snn.ConductanceBackground(cell.I, channel(2), channel(9)),
        ],
        modulation=[snn.ConductanceSchedule((cell.E, cell.I), 2, 6, end_scale=1.5)],
    )
    return snn.compile(net, simulation=simulation).write(tmp_path / "combined.bundle")


def test_combined_bundle_authenticates_dual_paths_and_owns_drive_flags(tmp_path):
    root = _write_combined_bundle(tmp_path)
    args = parse_args(["sim", "--bundle", str(root)])
    assert args.w_in == [0.01, 0.001]
    assert args.w_in_i == [0.02, 0.002]
    manifest, graph = load_graph_bundle(root)
    assert (
        load_simulation_recipe(root, manifest, graph)["schema"]
        == "snnlang.simulation/v1"
    )
    with pytest.raises(SystemExit):
        parse_args(["sim", "--bundle", str(root), "--independent-drive", "10", "0.1"])


def test_combined_input_realization_is_reproducible_private_and_shared(tmp_path):
    root = _write_combined_bundle(tmp_path)
    manifest, graph = load_graph_bundle(root)
    recipe = load_simulation_recipe(root, manifest, graph)
    kwargs = dict(
        seed=23,
        dt=0.25,
        t_steps=400,
        e_id="circuit_E",
        i_id="circuit_I",
        n_e=400,
        n_i=100,
        input_size=400,
    )
    first = realize_simulation_inputs(recipe, **kwargs)
    second = realize_simulation_inputs(recipe, **kwargs)
    assert first.retained.keys() == second.retained.keys()
    for key in first.retained:
        assert np.array_equal(first.retained[key], second.retained[key])
    shared = first.retained["input_excitatory_e_shared"]
    private = first.retained["input_excitatory_e_private"]
    assert np.array_equal(shared[:, 0], shared[:, -1])
    assert not np.array_equal(private[:, 0], private[:, -1])
    assert not np.array_equal(
        first.retained["input_excitatory_e_shared"],
        first.retained["input_inhibitory_e_shared"],
    )


def test_weather_realization_separates_afferents_and_local_shared_groups():
    channel = {
        "private": {
            "kind": "shot_noise",
            "rate_hz": 200,
            "amplitude": 0.02,
            "tau_ms": 2,
        },
        "shared": {
            "kind": "grouped_shot_noise",
            "rate_hz": 500,
            "amplitude": 0.01,
            "tau_ms": 2,
            "group_size": 4,
        },
        "heterogeneity": {
            "rate": {"kind": "constant", "value": 1},
            "amplitude": {"kind": "constant", "value": 1},
        },
    }
    recipe = {
        "weather": {"kind": "stationary_lognormal", "tau_ms": 50, "std_fraction": 0.2},
        "afferent_wave": {
            "kind": "smooth_transient",
            "onset_ms": 100,
            "peak_ms": 200,
            "plateau_end_ms": 200,
            "offset_ms": 300,
            "baseline_scale": 1,
            "peak_scale": 2,
            "shared_peak_scale": 3,
        },
        "spike_sources": [
            {
                "kind": "correlated_poisson_afferents",
                "input_e": "e.value",
                "input_i": "i.value",
                "shared_rate_hz": 20,
                "e_private_rate_hz": 30,
                "i_private_rate_hz": 40,
            }
        ],
        "backgrounds": [
            {
                "target": "E",
                "excitatory": channel,
                "inhibitory": channel,
            }
        ],
        "modulation": [],
    }

    realized = realize_simulation_inputs(
        recipe,
        seed=4,
        dt=1,
        t_steps=400,
        e_id="E",
        i_id="I",
        n_e=12,
        n_i=3,
        input_size=12,
    )

    assert realized.input_spikes_i is not None
    assert not np.array_equal(realized.input_spikes, realized.input_spikes_i)
    shared = realized.retained["input_afferent_shared"]
    assert np.all(realized.retained["input_structured_spikes_e"] >= shared)
    assert np.all(realized.retained["input_structured_spikes_i"] >= shared)
    weather = realized.retained["input_weather_scale"]
    assert np.all(np.isfinite(weather)) and np.all(weather > 0)
    wave = realized.retained["input_afferent_scale"]
    assert wave[0] == pytest.approx(1)
    assert wave[200] == pytest.approx(2)
    assert wave[300] == pytest.approx(1)
    shared_wave = realized.retained["input_afferent_shared_scale"]
    assert shared_wave[200] == pytest.approx(3)
    grouped = realized.retained["input_excitatory_e_shared"]
    assert np.array_equal(grouped[:, 0], grouped[:, 3])
    assert not np.array_equal(grouped[:, 0], grouped[:, 4])


def test_bundle_transition_builds_smooth_weight_schedule(tmp_path):
    baseline = _write_transition_bundle(tmp_path, "baseline", (0.4, 0.12))
    target = _write_transition_bundle(tmp_path, "target", (4.34, 1.302))
    args = SimpleNamespace(
        bundle=str(baseline),
        transition_bundle=str(target),
        transition_start_ms=20.0,
        transition_end_ms=60.0,
        t_ms=100.0,
    )
    schedules, time_ms, changed = _bundle_transition_schedule(args, 0.25, 400)
    assert changed == ["w_ee"]
    assert schedules["w_ee"][0] == pytest.approx(1.0)
    assert schedules["w_ee"][240] == pytest.approx(10.85)
    assert schedules["w_ee"][160] == pytest.approx((1.0 + 10.85) / 2)
    assert schedules["w_ei"].min() == schedules["w_ei"].max() == 1.0
    assert time_ms[-1] == pytest.approx(99.75)


def test_build_config_applies_cli_seed():
    args = parse_args(["sim", "--seed", "29"])
    built = config.build_config(args)
    assert built.seed == 29


def test_training_bundle_applies_graph_and_recipe(tmp_path):
    root = _write_bundle(tmp_path)
    args = parse_args(
        [
            "train",
            "--bundle",
            str(root),
            "--max-samples",
            "128",
            "--batch-size",
            "32",
        ]
    )
    assert args.model == "ping"
    assert args.dataset == "mnist"
    assert args.n_hidden == [256]
    assert args.dt == pytest.approx(0.1)
    assert args.readout_mode == "mem-mean"
    assert args.lr == pytest.approx(1e-3)
    assert args.weight_decay == pytest.approx(1e-4)
    assert args.epochs == 20
    assert args.max_samples == 128
    assert args.batch_size == 32


def test_training_recipe_flags_cannot_override_bundle(tmp_path):
    root = _write_bundle(tmp_path)
    with pytest.raises(SystemExit) as error:
        parse_args(["train", "--bundle", str(root), "--epochs", "1"])
    assert error.value.code == 2


def test_training_bundle_requires_authenticated_recipe(tmp_path):
    root = _write_bundle(tmp_path)
    (root / "training.json").unlink()
    with pytest.raises(SystemExit) as error:
        parse_args(["train", "--bundle", str(root)])
    assert error.value.code == 2


def test_training_recipe_rejects_unsupported_parameter_scope(tmp_path):
    root = _write_bundle(tmp_path)
    manifest, graph = load_graph_bundle(root)
    recipe = load_training_recipe(root, manifest, graph)
    recipe["parameter_groups"][0]["parameters"] = ["classifier_projection.weight"]
    recipe["parameter_groups"][1]["parameters"].append("sensory_ping_input.weight")
    with pytest.raises(BundleCompatibilityError, match="input/readout"):
        translate_training_v1(graph, recipe)


def _build_from_args(args):
    M.N_IN = args.n_in
    config.set_sim_dt(args.dt, getattr(args, "t_ms", 1.2))
    return config.build_net(
        args.model,
        refractory_e_ms=args.refractory_e_ms,
        refractory_i_ms=args.refractory_i_ms,
        refractory_policy=args.refractory_policy,
        w_in=args.w_in,
        w_in_initial_zero_fraction=args.w_in_initial_zero_fraction,
        w_ei=args.w_ei,
        w_ie=args.w_ie,
        ei_strength=args.ei_strength,
        ei_ratio=args.ei_ratio,
        recurrent_initial_zero_fraction=args.recurrent_initial_zero_fraction,
        hidden_sizes=args.n_hidden,
        readout_mode=args.readout_mode,
    )


def test_bundle_and_legacy_build_identical_cobanet(tmp_path):
    root = _write_bundle(tmp_path)
    bundle_args = parse_args(["sim", "--bundle", str(root)])
    legacy_args = parse_args([*_legacy_argv(), "--n-in", "784"])
    torch.manual_seed(17)
    bundle_net = _build_from_args(bundle_args)
    torch.manual_seed(17)
    legacy_net = _build_from_args(legacy_args)
    assert type(bundle_net) is type(legacy_net)
    assert sum(p.numel() for p in bundle_net.parameters()) == sum(
        p.numel() for p in legacy_net.parameters()
    )
    assert bundle_net.state_dict().keys() == legacy_net.state_dict().keys()
    for name, value in bundle_net.state_dict().items():
        torch.testing.assert_close(value, legacy_net.state_dict()[name], rtol=0, atol=0)


def _named_trainable(net):
    return {
        name: param for name, param in net.named_parameters() if param.requires_grad
    }


def _assert_tensor_maps_equal(stage, left, right):
    assert left.keys() == right.keys(), (
        f"{stage}: key mismatch "
        f"left_only={sorted(left.keys() - right.keys())} "
        f"right_only={sorted(right.keys() - left.keys())}"
    )
    for name in left:
        l_val = left[name]
        r_val = right[name]
        assert l_val.shape == r_val.shape, (
            f"{stage}: shape mismatch for {name}: {l_val.shape} != {r_val.shape}"
        )
        assert l_val.dtype == r_val.dtype, (
            f"{stage}: dtype mismatch for {name}: {l_val.dtype} != {r_val.dtype}"
        )
        torch.testing.assert_close(
            l_val,
            r_val,
            rtol=0,
            atol=0,
            msg=lambda msg, name=name, stage=stage: (
                f"{stage}: first divergent tensor {name}\n{msg}"
            ),
        )


def test_bundle_and_legacy_one_step_training_are_exactly_equivalent(tmp_path):
    root = _write_bundle(tmp_path)
    bundle_args = parse_args(["train", "--bundle", str(root), "--t-ms", "1.2"])
    legacy_args = parse_args(
        [
            *_legacy_argv("train"),
            "--t-ms",
            "1.2",
            "--lr",
            str(bundle_args.lr),
            "--weight-decay",
            str(bundle_args.weight_decay),
        ]
    )
    legacy_args.w_ei = [0.5, 0.05]
    legacy_args.w_ie = [1.0, 0.1]
    legacy_args.n_in = 784

    torch.manual_seed(123)
    bundle_net = _build_from_args(bundle_args)
    torch.manual_seed(123)
    legacy_net = _build_from_args(legacy_args)

    _assert_tensor_maps_equal(
        "initial state_dict",
        bundle_net.state_dict(),
        legacy_net.state_dict(),
    )

    bundle_trainable = _named_trainable(bundle_net)
    legacy_trainable = _named_trainable(legacy_net)
    assert set(bundle_trainable) == {"W_ff.0", "W_ff.1"}
    assert set(bundle_trainable) == set(legacy_trainable)
    assert {
        name for name, param in bundle_net.named_parameters() if not param.requires_grad
    } == {
        name for name, param in legacy_net.named_parameters() if not param.requires_grad
    }

    encoded_spikes = torch.zeros(12, 4, 784)
    encoded_spikes[0::3, :, 0:12] = 1.0
    encoded_spikes[1::3, :, 100:112] = 1.0
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    bundle_logits = bundle_net(input_spikes=encoded_spikes)
    legacy_logits = legacy_net(input_spikes=encoded_spikes)
    torch.testing.assert_close(
        bundle_logits,
        legacy_logits,
        rtol=0,
        atol=0,
        msg=lambda msg: f"forward logits diverged\n{msg}",
    )

    bundle_loss = F.cross_entropy(bundle_logits, labels)
    legacy_loss = F.cross_entropy(legacy_logits, labels)
    torch.testing.assert_close(
        bundle_loss,
        legacy_loss,
        rtol=0,
        atol=0,
        msg=lambda msg: f"cross-entropy loss diverged\n{msg}",
    )

    bundle_loss.backward()
    legacy_loss.backward()
    _assert_tensor_maps_equal(
        "gradients",
        {name: param.grad for name, param in bundle_trainable.items()},
        {name: param.grad for name, param in legacy_trainable.items()},
    )

    bundle_opt = torch.optim.AdamW(
        bundle_trainable.values(),
        lr=bundle_args.lr,
        weight_decay=bundle_args.weight_decay,
    )
    legacy_opt = torch.optim.AdamW(
        legacy_trainable.values(),
        lr=legacy_args.lr,
        weight_decay=legacy_args.weight_decay,
    )
    bundle_opt.step()
    legacy_opt.step()

    _assert_tensor_maps_equal(
        "post-AdamW state_dict",
        bundle_net.state_dict(),
        legacy_net.state_dict(),
    )

    bundle_opt_state = bundle_opt.state_dict()
    legacy_opt_state = legacy_opt.state_dict()
    assert bundle_opt_state["param_groups"] == legacy_opt_state["param_groups"]
    for param_idx, state in bundle_opt_state["state"].items():
        other = legacy_opt_state["state"][param_idx]
        _assert_tensor_maps_equal(
            f"AdamW optimizer state for parameter {param_idx}",
            state,
            other,
        )


def test_bundle_digest_is_authenticated(tmp_path):
    root = _write_bundle(tmp_path)
    graph_path = root / "graph.json"
    graph = json.loads(graph_path.read_text())
    graph["name"] = "tampered"
    graph_path.write_text(json.dumps(graph))
    with pytest.raises(BundleCompatibilityError, match="digest"):
        load_graph_bundle(root)


def test_structural_cli_override_is_rejected(tmp_path):
    root = _write_bundle(tmp_path)
    with pytest.raises(SystemExit) as error:
        parse_args(["sim", "--bundle", str(root), "--dt", "0.5"])
    assert error.value.code == 2


def test_bundle_path_is_not_inherited_from_legacy_load_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"bundle": "stale.bundle"}))
    with pytest.raises(SystemExit) as error:
        parse_args(["sim", "--load-config", str(config_path)])
    assert error.value.code == 2


def test_unsupported_graph_fails_with_capability_error(tmp_path):
    bundle = ping_classifier()
    bundle.graph["populations"].append(
        {
            "id": "extra",
            "size": 4,
            "neuron": {"kind": "lif"},
            "spiking": True,
            "group": None,
        }
    )
    with pytest.raises(BundleCompatibilityError, match="exactly two"):
        translate_cobanet_v1(bundle.graph)


def test_bundle_rejects_input_axis_order_that_backend_cannot_consume():
    bundle = ping_classifier()
    bundle.graph["inputs"][0]["shape"] = ["batch", "time", 784]
    with pytest.raises(BundleCompatibilityError, match="time.*batch.*channels"):
        translate_cobanet_v1(bundle.graph)


def test_bundle_cli_smoke_preserves_artifact_contract(tmp_path):
    root = _write_bundle(tmp_path)
    out = tmp_path / "run"
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "snnlab.sim",
            "sim",
            "--bundle",
            str(root),
            "--t-ms",
            "2",
            "--n-batch",
            "2",
            "--out-dir",
            str(out),
            "--wipe-dir",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PINGLAB_NO_COMPILE": "1"},
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert (out / "metrics.json").is_file()
    assert (out / "config.json").is_file()
    assert (out / "run.sh").is_file()
    config_data = json.loads((out / "config.json").read_text())
    assert config_data["bundle"] == str(root)
    assert config_data["n_hidden"] == [256]
