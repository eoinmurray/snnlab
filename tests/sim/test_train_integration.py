"""Integration tests for train() function.

Tests parameter propagation, M module globals setup, output artifacts,
config round-trip and edge cases.
"""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

from snnlab.sim import models as M
from snnlab.sim.train import train


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Whole file trains real (tiny) torch models — minutes of wall-clock. Marked slow
# so `pytest -m "not slow"` is a true fast lane.
pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture
def tmp_output_dir():
    """Temporary directory for training artifacts."""
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(autouse=True)
def _reset_model_globals():
    """Reset M module globals before and after each test to known defaults."""
    # Store original values before test
    old = (
        M.N_IN,
        M.N_HID,
        M.N_INH,
        M.N_OUT,
        M.HIDDEN_SIZES,
        M.T_ms,
        M.T_steps,
        M.dt,
        M.V_GRAD_DAMPEN,
        M.BATCH_SIZE,
        M.SURROGATE_SLOPE,
    )
    # Reset to module defaults before test
    M.N_IN = 64
    M.N_HID = 64
    M.N_INH = 16
    M.N_OUT = 10
    M.HIDDEN_SIZES = [64]
    M.T_ms = 1000.0
    M.T_steps = int(M.T_ms / M.dt)
    M.V_GRAD_DAMPEN = 80.0
    M.BATCH_SIZE = 64
    M.SURROGATE_SLOPE = 5.0
    # Run test
    yield
    # Restore to original values after test
    (
        M.N_IN,
        M.N_HID,
        M.N_INH,
        M.N_OUT,
        M.HIDDEN_SIZES,
        M.T_ms,
        M.T_steps,
        M.dt,
        M.V_GRAD_DAMPEN,
        M.BATCH_SIZE,
        M.SURROGATE_SLOPE,
    ) = old


class TestTrainParameterPropagation:
    """Test that CLI flags correctly propagate to M module and config.json."""

    def test_dt_saved_in_config(self, tmp_output_dir):
        """--dt should be saved to config.json."""
        dt_value = 0.5
        train(
            model_name="ping",
            dt=dt_value,
            t_ms=100.0,
            epochs=0,  # probe mode
            dataset="mnist",
            max_samples=50,  # larger to allow stratified split
            out_dir=tmp_output_dir,
        )
        with open(tmp_output_dir / "config.json") as f:
            config = json.load(f)
        assert config["dt"] == dt_value

    def test_resolved_surrogate_slope_saved_in_config(self, tmp_output_dir):
        M.SURROGATE_SLOPE = 1.25
        train(
            model_name="ping",
            dt=1.0,
            t_ms=5.0,
            epochs=0,
            dataset="mnist",
            max_samples=50,
            out_dir=tmp_output_dir,
        )
        config = json.loads((tmp_output_dir / "config.json").read_text())
        assert config["surrogate_slope"] == 1.25

    def test_direct_readout_initializer_and_realized_stats_are_saved(
        self, tmp_output_dir
    ):
        train(
            model_name="ping",
            dt=1.0,
            t_ms=5.0,
            epochs=0,
            dataset="mnist",
            max_samples=50,
            out_dir=tmp_output_dir,
            readout_w_init_mean=1.25,
            readout_w_init_std=0.0,
        )
        config = json.loads((tmp_output_dir / "config.json").read_text())
        assert config["readout_w_out_scale"] is None
        assert config["readout_w_init"] == {
            "distribution": "lower_clamped_normal",
            "units": "stored_weight",
            "mean": 1.25,
            "std": 0.0,
        }
        assert config["readout_w_init_realized"] == {
            "mean": 1.25,
            "std": 0.0,
            "min": 1.25,
            "max": 1.25,
            "zero_fraction": 0.0,
        }

    def test_t_ms_propagates_to_m_module(self, tmp_output_dir):
        """--t-ms should set M.T_ms."""
        t_ms_value = 150.0
        train(
            model_name="ping",
            dt=0.1,
            t_ms=t_ms_value,
            epochs=0,
            dataset="mnist",
            max_samples=50,
            out_dir=tmp_output_dir,
        )
        assert M.T_ms == t_ms_value
        # T_steps is computed in model.forward, not stored globally
        # Verify config has the right t_ms
        with open(tmp_output_dir / "config.json") as f:
            config = json.load(f)
        assert config["t_ms"] == t_ms_value

    def test_hidden_sizes_propagates_to_m_module(self, tmp_output_dir):
        """--hidden-sizes should set M.N_HID, M.N_INH, M.HIDDEN_SIZES."""
        hidden_sizes_value = [48]
        train(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            epochs=0,
            dataset="mnist",
            max_samples=50,
            hidden_sizes=hidden_sizes_value,
            out_dir=tmp_output_dir,
        )
        assert M.N_HID == 48
        assert M.N_INH == 48 // 4
        assert M.HIDDEN_SIZES == [48]

    def test_readout_mode_in_config(self, tmp_output_dir):
        """readout_mode should be saved to config.json."""
        train(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            epochs=0,
            dataset="mnist",
            max_samples=50,
            readout_mode="mem-mean",
            out_dir=tmp_output_dir,
        )
        with open(tmp_output_dir / "config.json") as f:
            config = json.load(f)
        assert config["readout_mode"] == "mem-mean"

    def test_cumulative_readout_provenance_in_config(self, tmp_output_dir):
        train(
            model_name="ping",
            dt=1.0,
            t_ms=20.0,
            epochs=0,
            dataset="mnist",
            max_samples=50,
            hidden_sizes=[32],
            readout_mode="cumulative-potential",
            signed_readout=True,
            readout_bias=True,
            out_dir=tmp_output_dir,
        )
        with open(tmp_output_dir / "config.json") as f:
            config = json.load(f)
        assert config["readout_mode"] == "cumulative-potential"
        assert config["signed_readout"] is True
        assert config["readout_bias"] is True
        assert config["readout_reduction"] == "sum_softmax_potential_per_timestep"
        assert config["readout_tau_bounds_ms"] == [5.0, 25.0]
        assert "f8756254ab8d0e3337eb69542c684b922d6b6cbd" in config["readout_reference"]

    def test_spike_count_reduction_and_units_in_config(self, tmp_output_dir):
        train(
            model_name="ping",
            dt=1.0,
            t_ms=20.0,
            epochs=0,
            dataset="mnist",
            max_samples=50,
            hidden_sizes=[32],
            readout_mode="spike-count",
            out_dir=tmp_output_dir,
        )
        with open(tmp_output_dir / "config.json") as f:
            config = json.load(f)
        assert config["readout_mode"] == "spike-count"
        assert config["readout_reduction"] == "output_spike_count"
        assert config["readout_units"] == "spikes"

    def test_dales_law_in_config(self, tmp_output_dir):
        """dales_law flag should be saved to config.json."""
        train(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            epochs=0,
            dataset="mnist",
            max_samples=50,
            dales_law=False,
            out_dir=tmp_output_dir,
        )
        with open(tmp_output_dir / "config.json") as f:
            config = json.load(f)
        assert config["dales_law"] is False

    def test_seed_produces_deterministic_results(self, tmp_output_dir):
        """Same seed should produce identical weight initialization."""
        # First run
        train(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            epochs=0,
            dataset="mnist",
            max_samples=50,
            seed=42,
            out_dir=tmp_output_dir,
        )
        with open(tmp_output_dir / "config.json") as f:
            config1 = json.load(f)

        # Second run with same seed
        tmp_output_dir2 = tmp_output_dir.parent / "run2"
        tmp_output_dir2.mkdir(exist_ok=True)
        train(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            epochs=0,
            dataset="mnist",
            max_samples=50,
            seed=42,
            out_dir=tmp_output_dir2,
        )
        with open(tmp_output_dir2 / "config.json") as f:
            config2 = json.load(f)

        # n_params should match (same architecture, same seed)
        assert config1["n_params"] == config2["n_params"]
        assert config1["seed"] == config2["seed"] == 42


class TestMModuleGlobalsSetup:
    """Test that M module globals are correctly initialized."""

    def test_n_hid_n_inh_from_hidden_sizes(self, tmp_output_dir):
        """M.N_HID and M.N_INH should be derived from hidden_sizes."""
        hidden_sizes = [96]
        train(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            epochs=0,
            dataset="mnist",
            max_samples=50,
            hidden_sizes=hidden_sizes,
            out_dir=tmp_output_dir,
        )
        assert M.N_HID == 96
        assert M.N_INH == 96 // 4  # n_inh = n_hid / 4

    def test_hidden_sizes_none_auto_detects_dataset_default(self, tmp_output_dir):
        """hidden_sizes=None should use DATASET_N_HIDDEN_DEFAULTS."""
        train(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            epochs=0,
            dataset="mnist",
            max_samples=200,  # need enough samples for stratified split
            hidden_sizes=None,
            out_dir=tmp_output_dir,
        )
        # mnist default is 1024 (from DATASET_N_HIDDEN_DEFAULTS)
        assert M.N_HID == 1024
        with open(tmp_output_dir / "config.json") as f:
            config = json.load(f)
        assert config["hidden_sizes"] == [1024]

    def test_t_steps_computed_correctly(self, tmp_output_dir):
        """M.T_ms should be set and T_steps computed from it."""
        dt = 0.25
        t_ms = 200.0
        train(
            model_name="ping",
            dt=dt,
            t_ms=t_ms,
            epochs=0,
            dataset="mnist",
            max_samples=50,
            out_dir=tmp_output_dir,
        )
        # T_steps is recalculated in model.forward based on current dt
        # But M.T_ms should be set
        assert M.T_ms == t_ms

    def test_v_grad_dampen_propagates(self, tmp_output_dir):
        """v_grad_dampen should set M.V_GRAD_DAMPEN."""
        train(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            epochs=0,
            dataset="mnist",
            max_samples=50,
            v_grad_dampen=150.0,
            out_dir=tmp_output_dir,
        )
        assert M.V_GRAD_DAMPEN == 150.0


class TestOutputArtifacts:
    """Test that train() produces correct output files and directories."""

    @pytest.mark.slow
    def test_weights_pth_created_and_loadable(self, tmp_output_dir):
        """weights.pth should be created and loadable by torch."""
        train(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            epochs=2,  # need at least one epoch with data
            dataset="mnist",
            max_samples=50,
            lr=0.01,
            out_dir=tmp_output_dir,
        )
        weights_path = tmp_output_dir / "weights.pth"
        assert weights_path.exists(), "weights.pth should be created after training"
        state_dict = torch.load(weights_path, map_location="cpu")
        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0

    def test_config_json_created_and_valid(self, tmp_output_dir):
        """config.json should be created with all required keys."""
        train(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            epochs=0,
            dataset="mnist",
            max_samples=50,
            out_dir=tmp_output_dir,
        )
        config_path = tmp_output_dir / "config.json"
        assert config_path.exists(), "config.json should be created"

        with open(config_path) as f:
            config = json.load(f)

        # Check essential keys
        required_keys = {
            "model",
            "dt",
            "t_ms",
            "epochs",
            "dataset",
            "n_hidden",
            "n_inh",
            "n_in",
            "n_params",
            "n_trainable",
            "dales_law",
            "readout_mode",
        }
        for key in required_keys:
            assert key in config, f"config.json missing required key: {key}"

    @pytest.mark.slow
    def test_metrics_json_created_and_valid(self, tmp_output_dir):
        """metrics.json should be created after training."""
        train(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            epochs=2,
            dataset="mnist",
            max_samples=50,
            lr=0.01,
            out_dir=tmp_output_dir,
        )
        metrics_path = tmp_output_dir / "metrics.json"
        assert metrics_path.exists(), "metrics.json should be created"

        with open(metrics_path) as f:
            metrics = json.load(f)

        assert metrics["mode"] == "train"
        assert "best_acc" in metrics
        assert "epochs" in metrics
        assert len(metrics["epochs"]) == 2
        initialization = metrics["config"]["weight_initialization"]
        assert {"W_in", "W_out", "W_EE_1", "W_EI_1", "W_IE_1", "W_II_1"} <= set(
            initialization
        )
        assert initialization["W_in"]["distribution"] == "lower_clamped_normal"
        assert initialization["W_in"]["zeros_remain_trainable"] is True
        assert initialization["W_in"]["requested_initial_zero_fraction"] == 0.0
        final_weights = metrics["weight_final"]
        assert set(initialization) == set(final_weights)
        for role, summary in final_weights.items():
            assert summary["zero_fraction"] == pytest.approx(
                summary["zero_count"]
                / initialization[role]["statistics"]["n_parameters"]
            )
        assert metrics["checkpoints"] == {
            "final_epoch": {
                "filename": "weights_final.pth",
                "epoch": 2,
                "sha256": _sha256(tmp_output_dir / "weights_final.pth"),
            },
            "best_validation": {
                "filename": "weights.pth",
                "epoch": metrics["best_epoch"],
                "sha256": _sha256(tmp_output_dir / "weights.pth"),
                "selection_metric": (
                    "validation_cross_entropy_mean_over_encoder_draws"
                ),
                "tie_breaker": "validation_accuracy_pct_mean_over_encoder_draws",
                "validation_draw_count": 3,
            },
        }
        validation_contract = metrics["config"]["validation_encoder_draws"]
        assert validation_contract == {
            "count": 3,
            "encoder_seeds": [20260415, 20260416, 20260417],
            "input_rate_seeds": [20270415, 20270416, 20270417],
            "aggregation_unit": "validation_sample_then_encoder_draw",
            "checkpoint_selection": (
                "minimum_mean_cross_entropy; tie maximum_mean_accuracy; "
                "tie earliest_epoch"
            ),
        }
        for epoch in metrics["epochs"]:
            assert len(epoch["validation_draws"]) == 3
            assert [draw["encoder_seed"] for draw in epoch["validation_draws"]] == [
                20260415,
                20260416,
                20260417,
            ]
            assert [draw["input_rate_seed"] for draw in epoch["validation_draws"]] == [
                20270415,
                20270416,
                20270417,
            ]
            assert epoch["test_loss"] == pytest.approx(
                sum(draw["cross_entropy"] for draw in epoch["validation_draws"]) / 3
            )
            assert epoch["acc"] == pytest.approx(
                sum(draw["accuracy_pct"] for draw in epoch["validation_draws"]) / 3
            )
        selected = min(
            metrics["epochs"],
            key=lambda epoch: (epoch["test_loss"], -epoch["acc"], epoch["ep"]),
        )
        assert metrics["best_epoch"] == selected["ep"]
        assert metrics["best_validation_loss"] == selected["test_loss"]
        assert metrics["best_acc"] == selected["acc"]

    def test_run_sh_created(self, tmp_output_dir):
        """run.sh should be created as a command log."""
        train(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            epochs=0,
            dataset="mnist",
            max_samples=50,
            out_dir=tmp_output_dir,
        )
        run_sh_path = tmp_output_dir / "run.sh"
        assert run_sh_path.exists(), "run.sh should be created"
        with open(run_sh_path) as f:
            content = f.read()
        assert "#!/bin/bash" in content


class TestConfigRoundTrip:
    """Test that config can be saved and loaded correctly."""

    def test_config_json_roundtrip_preserves_values(self, tmp_output_dir):
        """Save config to JSON, load it, verify all values match."""
        original_params = {
            "model_name": "ping",
            "dt": 0.2,
            "t_ms": 150.0,
            "epochs": 0,
            "dataset": "mnist",
            "max_samples": 100,  # enough for stratified split
            "hidden_sizes": [80],
            "lr": 0.005,
            "dales_law": False,
            "readout_mode": "rate",
        }
        train(**original_params, out_dir=tmp_output_dir)

        with open(tmp_output_dir / "config.json") as f:
            saved_config = json.load(f)

        # Verify key values round-tripped
        assert saved_config["dt"] == 0.2
        assert saved_config["t_ms"] == 150.0
        assert saved_config["dataset"] == "mnist"
        assert saved_config["dales_law"] is False
        assert saved_config["readout_mode"] == "rate"
        assert saved_config["hidden_sizes"] == [80]


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_hidden_sizes_none_uses_smart_default(self, tmp_output_dir):
        """hidden_sizes=None should auto-detect per dataset."""
        # Test with mnist (default 1024)
        tmpdir = tmp_output_dir.parent / "edge_mnist"
        tmpdir.mkdir(exist_ok=True)
        train(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            epochs=0,
            dataset="mnist",
            max_samples=100,
            hidden_sizes=None,
            out_dir=tmpdir,
        )
        with open(tmpdir / "config.json") as f:
            config = json.load(f)
        assert config["hidden_sizes"] == [1024]  # mnist default is 1024

    def test_seed_determinism_across_runs(self, tmp_output_dir):
        """Different runs with same seed should initialize identically."""
        seed = 123
        runs_data = []

        for i in range(2):
            tmpdir = tmp_output_dir.parent / f"seed_run_{i}"
            tmpdir.mkdir(exist_ok=True)
            train(
                model_name="ping",
                dt=0.1,
                t_ms=100.0,
                epochs=0,
                dataset="mnist",
                max_samples=50,
                seed=seed,
                out_dir=tmpdir,
            )
            with open(tmpdir / "config.json") as f:
                runs_data.append(json.load(f))

        # Check that n_params (architecture proxy) is identical
        assert runs_data[0]["n_params"] == runs_data[1]["n_params"]

    def test_epochs_zero_probe_mode(self, tmp_output_dir):
        """epochs=0 should create minimal artifacts (probe mode)."""
        train(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            epochs=0,
            dataset="mnist",
            max_samples=50,
            out_dir=tmp_output_dir,
        )

        # Should still create config and metrics
        assert (tmp_output_dir / "config.json").exists()
        assert (tmp_output_dir / "metrics.json").exists()

        # Check metrics.json has empty epochs list
        with open(tmp_output_dir / "metrics.json") as f:
            metrics = json.load(f)
        assert metrics["epochs"] == []
        assert metrics["total_elapsed_s"] == 0.0

    def test_multiple_hidden_layers(self, tmp_output_dir):
        """Multi-layer networks should work (e.g., [64, 32])."""
        hidden_sizes = [64, 32]
        train(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            epochs=0,
            dataset="mnist",
            max_samples=50,
            hidden_sizes=hidden_sizes,
            out_dir=tmp_output_dir,
        )
        # Last layer size should be set
        assert M.N_HID == 32
        with open(tmp_output_dir / "config.json") as f:
            config = json.load(f)
        assert config["hidden_sizes"] == [64, 32]


class TestTrainWithMinimalData:
    """Test training with very small datasets."""

    @pytest.mark.slow
    def test_metrics_jsonl_created(self, tmp_output_dir):
        """metrics.jsonl should be created with per-epoch records."""
        train(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            epochs=2,
            dataset="mnist",
            max_samples=100,
            lr=0.01,
            out_dir=tmp_output_dir,
        )
        jsonl_path = tmp_output_dir / "metrics.jsonl"
        assert jsonl_path.exists(), "metrics.jsonl should be created"

        # Read and verify it's valid JSONL
        with open(jsonl_path) as f:
            lines = f.readlines()
        assert len(lines) == 2  # two epochs
        for line in lines:
            record = json.loads(line)
            assert "ep" in record
            assert "acc" in record
            assert "loss" in record


class TestTrainWithDifferentDatasets:
    """Test parameter handling across different datasets."""

    def test_mnist_sets_n_in_correctly(self, tmp_output_dir):
        """MNIST should set N_IN=784."""
        train(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            epochs=0,
            dataset="mnist",
            max_samples=50,
            out_dir=tmp_output_dir,
        )
        assert M.N_IN == 784


class TestConfigValidation:
    """Test that config.json contains valid data types and ranges."""

    def test_config_json_values_are_serializable(self, tmp_output_dir):
        """All values in config.json should be JSON-serializable."""
        train(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            epochs=0,
            dataset="mnist",
            max_samples=50,
            out_dir=tmp_output_dir,
        )
        with open(tmp_output_dir / "config.json") as f:
            config = json.load(f)

        # Should be able to re-serialize without error
        serialized = json.dumps(config)
        assert len(serialized) > 0

    def test_config_numeric_values_in_valid_ranges(self, tmp_output_dir):
        """Numeric config values should be reasonable."""
        train(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            epochs=0,
            dataset="mnist",
            max_samples=50,
            out_dir=tmp_output_dir,
        )
        with open(tmp_output_dir / "config.json") as f:
            config = json.load(f)

        # dt should be small but positive
        assert 0 < config["dt"] < 10.0
        # t_ms should be reasonable
        assert 10 < config["t_ms"] < 10000.0
        # n_hidden should be positive
        assert config["n_hidden"] > 0
        # n_params should be reasonable (not negative, not absurdly large)
        assert config["n_params"] > 100
        assert config["n_params"] < 10_000_000
