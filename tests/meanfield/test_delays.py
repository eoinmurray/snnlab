"""Analytic delayed forcing and method-of-steps references, independent of gains."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from snnlab import meanfield as mf
from snnlab.lang.compilation.bundles import Bundle
from snnlab.meanfield import closures


def forcing_model(delays=(0.2, 0.7)):
    ctx = closures.BuildContext({}, {}, {}, {}, {}, "dynamic", "explicit")
    rate = ctx.state("rate", unit="Hz")
    ctx.derivative(rate, 1)
    for i, delay in enumerate(delays):
        state = ctx.state(f"g{i}", unit="uS")
        delayed = ctx.delay(rate, delay, name=f"path{i}.delayed_rate")
        ctx.derivative(state, -state + delayed)
        ctx.projection_delays[f"path{i}"] = delay
    return ctx.build()


def test_analytic_distinct_delays_shorter_than_observation_interval(tmp_path):
    model = forcing_model()
    initial = dict.fromkeys(model.states, 0)
    trajectory = mf.integrate(
        model,
        duration=3,
        initial=initial,
        history=mf.ConstantHistory(initial),
        sample_interval=0.5,
        max_step=0.3,
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(trajectory.state["rate"], trajectory.time_ms, atol=1e-12)
    for i, delay in enumerate((0.2, 0.7)):
        s = np.maximum(trajectory.time_ms - delay, 0)
        np.testing.assert_allclose(
            trajectory.state[f"g{i}"], s - 1 + np.exp(-s), atol=1e-10
        )
    saved = json.loads(trajectory.write(tmp_path / "trajectory.json").read_text())
    assert saved["integration"]["history"]["kind"] == "constant"
    assert saved["integration"]["solver"] == "method_of_steps"
    assert saved["integration"]["max_step_ms"] == 0.3
    assert saved["integration"]["delays_ms"] == {"path0": 0.2, "path1": 0.7}
    document = json.loads(model.write(tmp_path / "model.json").read_text())
    assert document["delays"]["required_history_interval_ms"] == [-0.7, 0]
    assert "t-0.2" in document["latex"]
    assert document["symbols"]["path0.delayed_rate"]["source"] == "rate"


def test_dde_feedback_polynomial_reference_and_convergence():
    # x'=x(t-1), x(t)=1 for t<=0. Method of steps gives truncated powers.
    ctx = closures.BuildContext({}, {}, {}, {}, {}, "dynamic", "explicit")
    x = ctx.state("x", unit="Hz")
    ctx.derivative(x, ctx.delay(x, 1, name="past_x"))
    model = ctx.build()

    def exact(t):
        return (
            1
            + t
            + np.maximum(t - 1, 0) ** 2 / 2
            + np.maximum(t - 2, 0) ** 3 / 6
            + np.maximum(t - 3, 0) ** 4 / 24
        )

    errors = []
    for rtol, step in ((1e-3, 0.8), (1e-10, 0.08)):
        result = mf.integrate(
            model,
            duration=4,
            initial={"x": 1},
            history=mf.ConstantHistory({"x": 1}),
            method="RK23",
            rtol=rtol,
            atol=rtol / 100,
            max_step=step,
            sample_interval=0.13,
        )
        errors.append(np.max(abs(result.state["x"] - exact(result.time_ms))))
    assert errors[1] < 1e-7
    assert errors[1] < errors[0] / 100


def test_nonconstant_history_and_its_units():
    model = forcing_model((1,))
    history = mf.History(
        lambda t: {"rate": t, "g0": 0},
        start=-1,
        description="rate(t)=t Hz for t in [-1,0]; zero conductance",
    )
    result = mf.integrate(
        model,
        duration=0.8,
        initial={"rate": 0, "g0": 0},
        history=history,
        sample_interval=0.1,
        method="DOP853",
        rtol=1e-11,
        atol=1e-13,
    )
    t = result.time_ms
    np.testing.assert_allclose(result.state["g0"], t - 2 + 2 * np.exp(-t), atol=1e-10)


@pytest.mark.parametrize(
    "history,error",
    [
        (None, "require History"),
        (
            mf.History(lambda t: {"rate": 0, "g0": 0}, start=-0.1, description="short"),
            "cover",
        ),
        (mf.ConstantHistory({"rate": 1, "g0": 0}), "agree"),
        (mf.ConstantHistory({"rate": 0}), "every retained"),
        (
            mf.History(
                lambda t: {"rate": np.nan, "g0": 0}, start=-1, description="invalid"
            ),
            "finite",
        ),
        (mf.History(lambda t: [0, 0], start=-1, description="unnamed"), "named"),
    ],
)
def test_invalid_histories(history, error):
    with pytest.raises(ValueError, match=error):
        mf.integrate(
            forcing_model((1,)),
            duration=1,
            initial={"rate": 0, "g0": 0},
            history=history,
        )


def test_history_quantity_validation():
    from snnlab import lang

    with pytest.raises(ValueError, match="expected Hz"):
        mf.integrate(
            forcing_model((1,)),
            duration=1,
            initial={"rate": 0, "g0": 0},
            history=mf.ConstantHistory({"rate": 0 * lang.ms, "g0": 0}),
        )


def test_exact_delivery_onset_preserves_state_and_does_not_shift_by_delay():
    ctx = closures.BuildContext({}, {}, {}, {}, {}, "dynamic", "explicit")
    rate = ctx.state("rate", unit="Hz")
    g = ctx.state("g", unit="uS")
    gain = ctx.parameter("gain", 1, unit="1")
    ctx.derivative(rate, 0)
    ctx.derivative(g, -g + gain * ctx.delay(rate, 2, name="past_rate"))
    model = ctx.build()
    initial = {"rate": 1, "g": 0.4}
    result = mf.integrate(
        model,
        duration=4,
        initial=initial,
        history=mf.ConstantHistory(initial),
        controls={"gain": mf.PiecewiseConstant((1.25,), (0, 1))},
        sample_interval=0.25,
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
        max_step=3,
    )
    t = result.time_ms
    expected = 0.4 * np.exp(-t) + np.where(
        t >= 1.25, 1 - np.exp(-np.maximum(t - 1.25, 0)), 0
    )
    np.testing.assert_allclose(result.state["g"], expected, atol=1e-9)
    assert result.state["g"][5] == pytest.approx(0.4 * np.exp(-1.25), abs=1e-10)


def test_external_parameter_controls_are_delayed():
    ctx = closures.BuildContext({}, {}, {}, {}, {}, "dynamic", "explicit")
    g = ctx.state("g", unit="uS")
    rate = ctx.parameter("rate", 0, unit="Hz")
    ctx.derivative(g, -g + ctx.delay(rate, 0.3, name="past_rate"))
    model = ctx.build()
    result = mf.integrate(
        model,
        duration=2,
        initial={"g": 0},
        history=mf.ConstantHistory({"g": 0}),
        controls={"rate": mf.PiecewiseConstant((0.2,), (0, 1))},
        sample_interval=0.1,
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.state["g"], 1 - np.exp(-np.maximum(result.time_ms - 0.5, 0)), atol=1e-8
    )


def test_zero_delay_keeps_original_ode_path():
    model = forcing_model((0,))
    result = mf.integrate(model, duration=3, initial=[0, 0], method="LSODA")
    reference = solve_ivp(
        lambda t, y: [1, -y[1] + y[0]],
        (0, 3),
        [0, 0],
        t_eval=result.time_ms,
        method="LSODA",
        rtol=1e-8,
        atol=1e-10,
        max_step=0.5,
    )
    np.testing.assert_array_equal(np.array(list(result.state.values())), reference.y)
    assert result.metadata["solver"] == "solve_ivp"


def graph_model(
    *, delays="explicit", modulation="reject", graph=None, external_filters="dynamic"
):
    if graph is None:
        graph = json.loads((Path(__file__).parent / "data/exp148.json").read_text())[
            "graph"
        ]
        graph = deepcopy(graph)
        for projection in graph["projections"]:
            projection.pop("modulation", None)
        graph["projections"][0]["delay"] = {"value": 0.1, "unit": "ms"}
        graph["projections"][1]["delay"] = {"value": 2, "unit": "ms"}
    return mf.reduce(
        Bundle(graph, None, {}, []),
        closure=closures.RateConductance(closures.NoisyLIF(sigma=4), tau_rate=5),
        weights=mf.MeanIncomingWeights(
            {p["id"]: 0.02 for p in graph["projections"]},
            provenance="test explicit couplings",
        ),
        inputs={"X": mf.StationaryRate(2.5)},
        delays=delays,
        modulation=modulation,
        external_filters=external_filters,
    )


def test_public_reduction_equilibrium_and_analysis_boundaries():
    model = graph_model()
    omitted = graph_model(delays="omit")
    eq = mf.find_equilibrium(model, guess=dict.fromkeys(model.states, 0.01))
    ode_eq = mf.find_equilibrium(omitted, guess=eq.state)
    np.testing.assert_allclose(
        list(eq.state.values()), list(ode_eq.state.values()), atol=1e-9
    )
    assert model.projection_delays["E_to_I"] == 2
    for operation in (
        lambda: mf.linearise(model, at=eq),
        lambda: mf.jacobian(model),
        lambda: model.numerical_jacobian,
        lambda: mf.continuation(model, "E_to_I.gain", [], guess=eq.state),
        lambda: mf.find_stability_crossings(model, "E_to_I.gain", []),
    ):
        with pytest.raises(ValueError, match="characteristic-root"):
            operation()
    with pytest.raises(ValueError, match="delayed histories"):
        mf.reduce_states(model)
    with pytest.raises(ValueError, match="require a history"):
        mf.evaluate(model, eq.state)
    with pytest.raises(ValueError, match="transmission delays"):
        graph_model(delays="reject")


def test_explicit_graph_modulation_translation():
    graph = json.loads((Path(__file__).parent / "data/exp148.json").read_text())[
        "graph"
    ]
    for projection in graph["projections"]:
        projection.pop("modulation", None)
    graph["projections"][0]["delay"] = {"value": 2, "unit": "ms"}
    graph["projections"][0]["modulation"] = {
        "kind": "step_gain",
        "at_ms": 0.123,
        "before": 0,
        "after": 1,
    }
    with pytest.raises(ValueError, match="graph modulation"):
        graph_model(graph=graph)
    model = graph_model(graph=graph, modulation="translate")
    control = model.controls["E_to_E.gain"]
    assert control(0.1499) == 0
    assert control(0.15 + 1e-15) == 1
    initial = dict.fromkeys(model.states, 0)
    initial["E.rate"] = 1
    result = mf.integrate(
        model,
        duration=0.3,
        initial=initial,
        history=mf.ConstantHistory(initial),
        sample_interval=0.05,
    )
    assert result.state["E_to_E.conductance"][2] == 0
    assert result.state["E_to_E.conductance"][-1] > 0


@pytest.mark.parametrize("delay", [-1, np.inf, np.nan])
def test_invalid_projection_delay(delay):
    graph = json.loads((Path(__file__).parent / "data/exp148.json").read_text())[
        "graph"
    ]
    graph["projections"][0]["delay"] = {"value": delay, "unit": "ms"}
    with pytest.raises(ValueError, match="nonnegative|finite"):
        graph_model(graph=graph)


def test_history_corner_is_propagated_and_state_is_continuous():
    ctx = closures.BuildContext({}, {}, {}, {}, {}, "dynamic", "explicit")
    x = ctx.state("x", unit="Hz")
    g = ctx.state("g", unit="uS")
    ctx.derivative(x, 0)
    ctx.derivative(g, ctx.delay(x, 1, name="past_x"))
    model = ctx.build()
    # Continuous history with a derivative corner at -0.4, arriving at t=0.6.
    history = mf.History(
        lambda t: {"x": max(t + 0.4, 0), "g": 0},
        start=-1,
        description="x=max(t+0.4,0), g=0",
        breakpoints=(-0.4,),
    )
    result = mf.integrate(
        model,
        duration=1.5,
        initial={"x": 0.4, "g": 0},
        history=history,
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
        sample_interval=0.1,
    )
    t = result.time_ms
    expected = np.where(t <= 1, np.maximum(t - 0.6, 0) ** 2 / 2, 0.08 + 0.4 * (t - 1))
    np.testing.assert_allclose(result.state["g"], expected, atol=1e-10)


def test_ode_control_at_final_endpoint_is_not_integrated_prematurely():
    ctx = closures.BuildContext({}, {}, {}, {}, {}, "dynamic", "reject")
    x = ctx.state("x", unit="Hz")
    gain = ctx.parameter("gain", 0, unit="Hz")
    ctx.derivative(x, gain)
    result = mf.integrate(
        ctx.build(),
        duration=1,
        initial={"x": 0},
        controls={"gain": mf.PiecewiseConstant((1,), (0, 1))},
        max_step=2,
    )
    np.testing.assert_array_equal(result.state["x"], 0)


def test_translation_keeps_graph_provenance_and_supported_schedule_samples():
    from snnlab._shared.gains import sample
    from snnlab.lang.compilation.serialization import digest
    from snnlab.meanfield.controls import graph_control

    specs = [
        {"kind": "constant_gain", "value": 0.7},
        {"kind": "step_gain", "at_ms": 0.123, "before": 0, "after": 1},
        {
            "kind": "piecewise_gain",
            "times_ms": [0.11, 0.12, 0.36],
            "values": [0, 1, 0.7, 0.2],
        },
        {
            "kind": "ramp_gain",
            "start_ms": 0.11,
            "end_ms": 0.36,
            "before": 0,
            "after": 1,
        },
    ]
    for spec in specs:
        control = graph_control(spec, 0.05)
        np.testing.assert_allclose(
            [control(t * 0.05) for t in range(10)],
            sample(spec, start_step=0, steps=10, dt_ms=0.05),
            atol=1e-15,
        )
    graph = json.loads((Path(__file__).parent / "data/exp148.json").read_text())[
        "graph"
    ]
    original = deepcopy(graph)
    model = graph_model(graph=graph, modulation="translate")
    assert model.provenance["graph_digest"] == digest(original)
    assert graph == original
    with pytest.raises(ValueError, match="unsupported graph modulation"):
        graph_control(
            {
                "kind": "sine_gain",
                "frequency_hz": 10,
                "depth": 0.1,
                "offset": 1,
                "phase": 0,
            },
            0.05,
        )


def test_explicit_external_delay_rejects_quasistatic_filter():
    with pytest.raises(ValueError, match="dynamic external"):
        graph_model(external_filters="quasistatic")


def test_noncommensurate_delays_and_short_final_interval():
    model = forcing_model((0.173, np.sqrt(2) / 10))
    initial = dict.fromkeys(model.states, 0)
    result = mf.integrate(
        model,
        duration=0.95,
        initial=initial,
        history=mf.ConstantHistory(initial),
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
        sample_interval=0.33,
    )
    for i, delay in enumerate((0.173, np.sqrt(2) / 10)):
        s = np.maximum(result.time_ms - delay, 0)
        np.testing.assert_allclose(
            result.state[f"g{i}"], s - 1 + np.exp(-s), atol=1e-10
        )


def test_only_delayed_external_controls_require_pre_zero_values():
    ctx = closures.BuildContext({}, {}, {}, {}, {}, "dynamic", "explicit")
    g = ctx.state("g", unit="uS")
    rate = ctx.parameter("rate", 1, unit="Hz")
    gain = ctx.parameter("gain", 1, unit="1")
    ctx.derivative(g, gain * ctx.delay(rate, 1, name="past_rate"))

    def delivery_gain(time):
        assert time >= 0
        return 1

    result = mf.integrate(
        ctx.build(),
        duration=1,
        initial={"g": 0},
        history=mf.ConstantHistory({"g": 0}),
        controls={"gain": delivery_gain},
    )
    np.testing.assert_allclose(result.state["g"], result.time_ms, atol=1e-10)


@pytest.mark.parametrize("duration", [1500, 5000])
def test_long_commensurate_delays_have_one_boundary_per_grid_point(duration):
    from snnlab.meanfield.integration import boundaries

    # The onset and history corner already lie on the 0.1 ms grid. Every route
    # to a given sum (twenty 0.1 ms hops or one 2 ms hop) must deduplicate.
    actual = boundaries(duration, (0.1, 2), (-0.1, 500))
    expected = np.arange(duration * 10 + 1, dtype=np.int64) / 10
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.integration
@pytest.mark.slow
def test_full_duration_delayed_integration_against_analytic_forcing():
    model = forcing_model((0.1, 2))
    initial = dict.fromkeys(model.states, 0)
    result = mf.integrate(
        model,
        duration=5000,
        initial=initial,
        history=mf.ConstantHistory(initial),
        breakpoints=(500,),
        sample_interval=500,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )
    assert result.metadata["segments"] == 50_000
    assert result.time_ms[-1] == 5000
    np.testing.assert_allclose(result.state["rate"], result.time_ms, atol=1e-7, rtol=0)
    for i, delay in enumerate((0.1, 2)):
        s = np.maximum(result.time_ms - delay, 0)
        np.testing.assert_allclose(
            result.state[f"g{i}"], s - 1 + np.exp(-s), atol=1e-7, rtol=0
        )


def test_boundary_grid_preserves_offset_events_and_non_grid_endpoint():
    from snnlab.meanfield.integration import boundaries

    actual = boundaries(1.03, (0.1, 0.2), (0.05,))
    expected = [*np.arange(21, dtype=np.int64) / 20, 1.03]
    np.testing.assert_array_equal(actual, expected)


def test_boundary_complexity_guard_still_rejects_distinct_events():
    from snnlab.meanfield.integration import boundaries

    with pytest.raises(ValueError, match="too many delay discontinuities"):
        boundaries(10_000, (0.1, 2), ())
    with pytest.raises(ValueError, match="boundary resolution"):
        boundaries(1, (1e-14,), ())
