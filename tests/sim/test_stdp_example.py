"""The STDP example must reproduce itself and support the claims its page makes."""

import runpy
from pathlib import Path

import numpy as np
import pytest

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/stdp"


@pytest.fixture(scope="module")
def example():
    return runpy.run_path(str(EXAMPLE / "example.py"))


@pytest.fixture(scope="module")
def short_runs(example):
    first = example["run_example"](learning_seconds=2, probe_seconds=1)
    second = example["run_example"](learning_seconds=2, probe_seconds=1)
    return first, second


def test_the_same_seed_reproduces_the_run_exactly(example, short_runs):
    first, second = short_runs
    for key in ("initial", "rates", "snapshots", "final", "rate_before", "rate_after"):
        np.testing.assert_array_equal(first[key], second[key])
    assert example["summarise"](first) == example["summarise"](second)


def test_recorded_weights_end_at_the_final_weights_within_the_bounds(
    example, short_runs
):
    run, _ = short_runs
    trials = len(example["EXPERIMENTS"])
    assert run["snapshots"].shape == (2, trials, example["N_EXC"])
    assert run["final"].shape == (trials, example["N_EXC"])
    np.testing.assert_array_equal(run["snapshots"][-1], run["final"])
    w_max = np.float32(example["W_MAX_US"])
    assert run["final"].min() >= 0 and run["final"].max() <= w_max
    # Every trial starts from the model's weights and learns its own copy.
    assert not np.array_equal(run["final"][0], run["final"][2])
    assert not np.array_equal(run["final"][0], run["initial"])


def test_the_learning_window_is_the_formula(example):
    lags_ms, change = example["learning_window"]()
    tau = example["TAU_MS"]
    a_plus, a_minus = example["A_PLUS_US"], example["A_MINUS_US"]
    expected = np.where(
        lags_ms > 0,
        a_plus * np.exp(-lags_ms / tau),
        -a_minus * np.exp(lags_ms / tau),
    )
    expected[lags_ms == 0] = 0  # same-step spikes do not pair
    np.testing.assert_allclose(change, expected, rtol=1e-12, atol=1e-18)


def test_the_figures_and_summary_are_written(
    example, short_runs, tmp_path, monkeypatch
):
    run, _ = short_runs
    monkeypatch.setitem(example["plot_window"].__globals__, "OUTPUT_DIR", tmp_path)
    example["plot_window"](run["window"])
    example["plot_learning"](run)
    assert (tmp_path / "stdp-window.png").stat().st_size > 0
    assert (tmp_path / "stdp-learning.png").stat().st_size > 0
    summary = example["summarise"](run)
    assert summary["learning_seconds"] == 2
    assert [e["label"] for e in summary["experiments"]] == [
        label for label, _, _ in example["EXPERIMENTS"]
    ]


@pytest.mark.slow
def test_the_documented_run_supports_the_page_claims(example):
    # The full documented run: the page's three claims, with margins.
    summary = example["summarise"](example["run_example"]())
    independent, correlated, fast = summary["experiments"]
    # The output rate is regulated: the 20 Hz/10 Hz output ratio shrinks.
    before = fast["output_rate_before_hz"] / independent["output_rate_before_hz"]
    after = fast["output_rate_after_hz"] / independent["output_rate_after_hz"]
    assert before > 3
    assert after < 2.5
    assert fast["output_rate_after_hz"] < 0.5 * fast["output_rate_before_hz"]
    # Correlated inputs win; the other inputs of the same neuron lose.
    assert correlated["mean_weight_first_half"] > 0.9
    assert correlated["mean_weight_second_half"] < 0.4
    assert 0.4 < independent["mean_weight_first_half"] < 0.6
    assert 0.4 < independent["mean_weight_second_half"] < 0.6
    # The distributions move towards the bounds.
    assert correlated["fraction_above_0.9"] > 0.4
    assert fast["fraction_below_0.1"] > 0.2
