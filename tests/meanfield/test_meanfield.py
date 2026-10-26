"""Independent exp148 evidence and invariants of the symbolic model API."""

import ast
import importlib.util
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import sympy as sp
from scipy.integrate import quad
from scipy.special import erfcx

from snnlab import lang
from snnlab import meanfield as mf
from snnlab.lang.compilation.bundles import Bundle
from snnlab.meanfield import closures

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/meanfield/example.py"
spec = importlib.util.spec_from_file_location("meanfield_example", EXAMPLE)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def load_reference():
    return json.loads((Path(__file__).parent / "data/exp148.json").read_text())


def test_example_has_four_functions_and_preserves_couplings():
    functions = [
        node.name
        for node in ast.walk(ast.parse(EXAMPLE.read_text()))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    assert functions == ["configuration", "build_model", "run", "plot"]
    assert "exp148" not in EXAMPLE.read_text()
    assert (
        example.configuration()["mean_incoming_weights_us"]
        == load_reference()["configuration"]["summed_weights_us"]
    )


def test_example_integrates_all_four_recurrent_conductances():
    model, _ = example.build_model()
    assert list(model.states) == [
        "E.rate",
        "I.rate",
        "E_to_E.conductance",
        "E_to_I.conductance",
        "I_to_E.conductance",
        "I_to_I.conductance",
    ]
    assert not model.reconstructions and not model.constraints
    initial = dict(zip(model.states, [12, 25, 0.3, 0.07, 0.11, 0.4], strict=True))
    derivative = dict(zip(model.states, mf.evaluate(model, initial), strict=True))
    for pathway in ("E_to_E", "E_to_I", "I_to_E", "I_to_I"):
        state = pathway + ".conductance"
        source_rate = initial[pathway.split("_to_")[0] + ".rate"]
        expected = (
            -initial[state] / model.defaults[pathway + ".tau"]
            + model.defaults[pathway + ".coupling"] * source_rate / 1000
        )
        assert derivative[state] == pytest.approx(expected)


@pytest.fixture(scope="module")
def models():
    full, control = example.build_model()
    reduced = mf.reduce_states(
        full, retain=["E_to_I.conductance", "I_to_E.conductance"]
    )
    return full, reduced, control


def test_frozen_exp148_equilibria_and_eigenvalues(models):
    full, model, _ = models
    assert len(full.states) == 6
    assert list(model.states) == [
        "E.rate",
        "I.rate",
        "E_to_I.conductance",
        "I_to_E.conductance",
    ]
    for row in load_reference()["reference"]["continuation"]:
        eq = mf.find_equilibrium(
            model, guess=row["state"], parameters={"X_to_I.gain": row["alpha"]}
        )
        np.testing.assert_allclose(list(eq.state.values()), row["state"], atol=1e-10)
        stability = mf.linearise(model, at=eq)
        expected = np.array(row["eigenvalues_real"]) + 1j * np.array(
            row["eigenvalues_imag"]
        )
        np.testing.assert_allclose(
            np.sort_complex(stability.eigenvalues),
            np.sort_complex(expected),
            atol=1e-10,
        )


def test_generated_jacobian_against_finite_differences(models):
    _, model, _ = models
    state = np.array([12, 25, 0.07, 0.11])
    numeric = np.empty((4, 4))
    for j, step in enumerate([1e-4, 1e-4, 1e-7, 1e-7]):
        delta = np.eye(4)[j] * step
        numeric[:, j] = (
            mf.evaluate(model, state + delta) - mf.evaluate(model, state - delta)
        ) / (2 * step)
    np.testing.assert_allclose(
        mf.linearise(model, at=state).jacobian, numeric, rtol=1e-6, atol=1e-7
    )
    assert isinstance(mf.jacobian(model), sp.MatrixBase)
    assert len(mf.equations(model)) == 4
    assert "\\begin{aligned}" in mf.latex(model)


def test_mathematical_notation_is_an_exact_presentation_of_the_model(models):
    for model in models[:2]:
        for display, computational in zip(model.display_rhs, model.rhs, strict=True):
            expression = display
            for _ in range(len(model.definitions) + 1):
                if not expression.free_symbols.intersection(model.definitions):
                    break
                expression = expression.subs(model.definitions, simultaneous=True)
            assert not expression.free_symbols.intersection(model.definitions)
            assert sp.simplify(sp.expand(expression) - sp.expand(computational)) == 0
        tex = mf.latex(model)
        assert r"\dot{r_{E}}" in tex
        assert r"g_{e}^{I}" in tex
        assert r"\Delta V_{\mathrm{exc},E}" in tex
        assert r"C_{E\to I}" in tex
        assert (
            "E.rate" not in tex and "conductance" not in tex and "coupling" not in tex
        )
        assert " &= " in tex
        assert r"I_{E} &=" in tex
        assert r"I_{E} &=" not in mf.latex(model, expanded=True)


def test_model_write_exports_individually_addressable_equations(models, tmp_path):
    _, model, _ = models
    path = model.write(tmp_path / "nested/model.json")
    data = json.loads(
        path.read_text(),
        parse_constant=lambda value: pytest.fail(f"nonstandard JSON value: {value}"),
    )
    assert data["schema"] == "snnlab.meanfield.model/v1"
    assert data["state_order"] == list(model.states)
    assert data["assumptions"] == list(model.assumptions)
    assert data["latex"] == mf.latex(model)
    assert data["states"]["E.rate"]["unit"] == "Hz"
    assert data["parameters"]["E.tau_rate"]["value"] == 20
    coupling = data["symbols"]["E_to_I.coupling"]
    assert coupling == {**data["parameters"]["E_to_I.coupling"], "kind": "parameter"}
    assert (
        coupling["definition_latex"]
        == r"C_{E\to I} = \frac{1}{N_{I}}\sum_{j,k}\mathbf{W}_{jk}^{E\to I}"
    )
    assert coupling["aggregation"]["target_count"] == 1000
    assert coupling["aggregation"]["source_count"] == 4000
    assert coupling["aggregation"]["includes_zero_edges"]
    assert coupling["components"]["N_I"]["value"] == 1000
    assert coupling["components"]["W"]["unit"] == "uS"
    assert coupling["components"]["W"]["indices"]["j"]["size"] == 4000
    assert "averaged over target neurons" in coupling["description"]
    assert data["states"]["E_to_I.conductance"]["bounds"] == [0, None]
    assert data["model_definition"]["rhs"]["E.rate"] == str(model.rhs[0])
    assert data["model_definition"]["functions"]["Phi_E"]["sigma_mv"] == 4
    assert data["model_definition"]["functions"]["Phi_E_slope"]["of"] == "Phi_E"
    assert data["model_definition"]["reconstructions"] == {
        name: str(expr) for name, expr in model.reconstructions.items()
    }
    for name in model.states:
        record = data["equations"][name]
        assert record["kind"] == "differential"
        assert record["latex"] == record["lhs_latex"] + " = " + record["rhs_latex"]
    assert data["equations"]["E.effective_current"]["kind"] == "definition"
    assert (
        data["symbols"]["E.effective_current"]["definition_latex"]
        == data["equations"]["E.effective_current"]["latex"]
    )
    assert data["symbols"]["E.effective_current"]["unit"] == "nA"
    assert data["symbols"]["E_to_E.conductance"]["kind"] == "derived"
    assert "E.effective_current" in data["equations"]["E.rate"]["dependencies"]
    assert "E_to_E.conductance" in data["equations"]
    # No execution or sampling is needed to reproduce the same document.
    previous = path.read_bytes()
    model.write(path)
    assert path.read_bytes() == previous


def test_failed_model_write_preserves_an_existing_file(models, tmp_path):
    path = tmp_path / "model.json"
    path.write_text("previous export")
    invalid = replace(models[1], function_definitions={"Phi_E": {"invalid": object()}})
    with pytest.raises(TypeError):
        invalid.write(path)
    assert path.read_text() == "previous export"
    assert list(tmp_path.iterdir()) == [path]


def test_custom_closure_can_describe_a_shorthand(tmp_path):
    ctx = closures.BuildContext({}, {}, {}, {}, {}, "dynamic", "reject")
    rate = ctx.state("rate", unit="Hz")
    target = ctx.parameter("target", 10, unit="Hz")
    ctx.describe(
        target,
        description="Prescribed target rate",
        definition_latex=r"r_0 = 10\,\mathrm{Hz}",
    )
    ctx.derivative(rate, target - rate)
    path = ctx.build().write(tmp_path / "custom.json")
    assert (
        json.loads(path.read_text())["symbols"]["target"]["description"]
        == "Prescribed target rate"
    )
    with pytest.raises(ValueError, match="declared"):
        ctx.describe(sp.Symbol("unknown"), description="Not a model symbol")


def test_full_and_reduced_vector_fields_agree_on_manifold(models):
    full, model, _ = models
    state = dict(zip(model.states, [12, 25, 0.07, 0.11], strict=True))
    parameters = {
        symbol: model.defaults[name] for name, symbol in model.parameters.items()
    }
    substitutions = {
        **parameters,
        **{model.states[name]: value for name, value in state.items()},
    }
    expanded = {
        **state,
        **{
            name: float(expr.subs(substitutions))
            for name, expr in model.reconstructions.items()
        },
    }
    original = dict(zip(full.states, mf.evaluate(full, expanded), strict=True))
    reduced = dict(zip(model.states, mf.evaluate(model, state), strict=True))
    np.testing.assert_allclose(
        [original[name] for name in model.states], list(reduced.values()), atol=1e-12
    )
    for name, expr in model.reconstructions.items():
        derivative = sum(
            float(sp.diff(expr, symbol).subs(substitutions)) * reduced[key]
            for key, symbol in model.states.items()
        )
        assert original[name] == pytest.approx(derivative, abs=1e-12)


def test_matched_decay_rejects_invalid_parameters_and_controls(models):
    full, model, _ = models
    with pytest.raises(ValueError, match="matching decay"):
        mf.reduce_states(mf.substitute(full, {"E_to_E.tau": 6}))
    with pytest.raises(ValueError, match="constraint"):
        mf.evaluate(model, [2, 6, 0, 0], parameters={"E_to_E.tau": 6})
    with pytest.raises(ValueError, match="nonzero"):
        mf.substitute(model, {"E_to_I.gain": 0})
    with pytest.raises(ValueError, match="controls cannot vary"):
        mf.integrate(
            model,
            duration=1,
            initial=[2, 6, 0, 0],
            controls={"E_to_E.gain": lambda t: t},
        )
    changed = mf.substitute(model, {"X_to_I.gain": 0.2})
    assert changed.defaults["X_to_I.gain"] == 0.2
    assert model.defaults["X_to_I.gain"] == 1


def test_noisy_lif_against_adaptive_quadrature():
    fixture = load_reference()
    graph = fixture["graph"]
    ctx = closures.BuildContext(graph, {}, {}, {}, {}, "dynamic", "reject")
    gain, maximum = closures.NoisyLIF(sigma=4 * lang.mV).bind(
        ctx, graph["populations"][0]
    )
    current = sp.Symbol("I")
    function = sp.lambdify(current, gain(current), modules=[ctx.functions, "numpy"])
    for value in [0, 0.1, 0.3, 0.6, 2.0]:
        mean = -70 + value / 0.0167
        integral = quad(lambda u: erfcx(-u), (-60 - mean) / 4, (-50 - mean) / 4)[0]
        expected = 1000 / (2 + (0.25 / 0.0167) * np.sqrt(np.pi) * integral)
        assert function(value) == pytest.approx(expected, rel=1e-10, abs=1e-12)
    assert maximum == 500


def test_realised_matrix_summaries_include_zeros_and_target_normalisation():
    graph = deepcopy(load_reference()["graph"])
    graph["populations"][0]["size"] = 2
    graph["populations"][1]["size"] = 3
    graph["projections"] = [graph["projections"][1]]
    weights = mf.RealisedWeights(
        {"E_to_I.weight": [[1, 2, 0], [0, 3, 0]]}, provenance="test effective matrix"
    )
    assert weights.resolve(graph) == {"E_to_I": 2}
    with pytest.raises(ValueError, match="exactly once"):
        mf.MeanIncomingWeights({}, provenance="missing").resolve(graph)


def test_custom_closure_uses_same_symbolic_and_numerical_pipeline():
    class LinearRate:
        def build(self, ctx):
            rate = ctx.state("E.rate", unit="Hz", bounds=(0, float("inf")))
            target = ctx.parameter("target", 10, unit="Hz")
            tau = ctx.parameter("tau", 5 * lang.ms, unit="ms")
            ctx.derivative(rate, (target - rate) / tau)
            ctx.assumptions.append("Custom linear relaxation")
            return ctx.build()

    net = lang.Network("custom")
    net.population("E", size=2, neuron=lang.COBA_LIF(tau_mem=20 * lang.ms))
    model = mf.reduce(
        net,
        closure=LinearRate(),
        weights=mf.MeanIncomingWeights({}, provenance="no projections"),
    )
    equilibrium = mf.find_equilibrium(model, guess={"E.rate": 1})
    assert equilibrium.state["E.rate"] == pytest.approx(10)
    np.testing.assert_allclose(mf.linearise(model, at=equilibrium).eigenvalues, [-0.2])
    trajectory = mf.integrate(model, duration=10, initial={"E.rate": 0})
    np.testing.assert_allclose(
        trajectory.state["E.rate"],
        10 * (1 - np.exp(-trajectory.time_ms / 5)),
        atol=1e-7,
    )


def test_exp117_style_cross_coupling_has_four_states():
    fixture = load_reference()
    graph = deepcopy(fixture["graph"])
    graph["projections"] = [
        p for p in graph["projections"] if p["id"] in ("E_to_I", "I_to_E")
    ]
    graph["inputs"] = []
    model = mf.reduce(
        Bundle(graph, None, {}, []),
        closure=closures.RateConductance(closures.NoisyLIF(sigma=4), {"E": 20, "I": 5}),
        weights=mf.MeanIncomingWeights(
            {"E_to_I": 1, "I_to_E": 2}, provenance="explicit cross coupling"
        ),
        applied_current={"E": 1 * lang.nA},
        delays="omit",
    )
    assert len(model.states) == 4
    derivative = mf.evaluate(model, [2, 6, 0.1, 0.2])
    assert derivative[2] == pytest.approx(-0.1 / 5 + 2 / 1000)
    assert derivative[3] == pytest.approx(-0.2 / 5 + 2 * 6 / 1000)


def test_graph_schedules_and_delays_require_explicit_handling():
    fixture = load_reference()
    kwargs = dict(
        closure=closures.RateConductance(closures.NoisyLIF(sigma=4), {"E": 20, "I": 5}),
        weights=mf.MeanIncomingWeights(
            fixture["configuration"]["summed_weights_us"], provenance="fixture"
        ),
        inputs={"X": mf.StationaryRate(2.5)},
    )
    bundle = Bundle(fixture["graph"], None, {}, [])
    with pytest.raises(ValueError, match="delays"):
        mf.reduce(bundle, **kwargs)
    with pytest.raises(ValueError, match="graph modulation"):
        mf.reduce(bundle, **kwargs, delays="omit")
    with pytest.raises(ValueError, match="expected Hz"):
        mf.StationaryRate(2 * lang.ms)


def test_composes_three_populations_and_inherits_synapse_default():
    net = lang.Network("three-population")
    populations = {
        name: net.population(name, size=2, neuron=lang.COBA_LIF(tau_mem=10 * lang.ms))
        for name in ("A", "B", "C")
    }
    for source, target in (("A", "B"), ("B", "C"), ("C", "A")):
        net.connect(
            populations[source].spikes,
            populations[target].excitatory,
            name=f"{source}_to_{target}",
            synapse=lang.AMPA(),
        )
    model = mf.reduce(
        net,
        closure=closures.RateConductance(closures.NoisyLIF(sigma=4), tau_rate=5),
        weights=mf.MeanIncomingWeights(
            {"A_to_B": 1, "B_to_C": 2, "C_to_A": 3}, provenance="explicit couplings"
        ),
    )
    assert len(model.states) == 6
    assert model.defaults["A_to_B.tau"] == 1
    state = {name: (10 if name.endswith("rate") else 0.1) for name in model.states}
    derivative = dict(zip(model.states, mf.evaluate(model, state), strict=True))
    assert derivative["A_to_B.conductance"] == pytest.approx(-0.1 + 0.01)
    with pytest.raises(ValueError, match="bounds"):
        mf.substitute(model, {"A_to_B.tau": 0})


def test_dynamic_external_filter_retains_its_transient():
    fixture = load_reference()
    graph = deepcopy(fixture["graph"])
    graph["projections"] = [p for p in graph["projections"] if p["id"] == "X_to_E"]
    model = mf.reduce(
        Bundle(graph, None, {}, []),
        closure=closures.RateConductance(closures.NoisyLIF(sigma=4), tau_rate=5),
        weights=mf.MeanIncomingWeights(
            {"X_to_E": 2}, provenance="explicit external coupling"
        ),
        inputs={"X": mf.StationaryRate(10)},
        delays="omit",
    )
    trajectory = mf.integrate(
        model, duration=20, initial={"E.rate": 0, "I.rate": 0, "X_to_E.conductance": 0}
    )
    expected = 5 * 2 * 10 / 1000 * (1 - np.exp(-trajectory.time_ms / 5))
    np.testing.assert_allclose(
        trajectory.state["X_to_E.conductance"], expected, atol=1e-9
    )


@pytest.mark.integration
def test_complete_exp148_example_against_frozen_results(tmp_path):
    result, trajectory = example.run(tmp_path)
    reference = load_reference()["reference"]
    crossing = result["crossings"][0]
    assert crossing["oscillatory"] and crossing["destabilizes_as_alpha_decreases"]
    assert crossing["alpha"] == pytest.approx(
        reference["crossings"][0]["alpha"], abs=1e-9
    )
    assert crossing["frequency_hz"] == pytest.approx(
        reference["crossings"][0]["frequency_hz"], abs=1e-7
    )
    indices = np.searchsorted(trajectory.time_ms, reference["trajectory"]["time_ms"])
    assert len(trajectory.state) == 6
    actual = np.array(
        [
            trajectory.state[name]
            for name in ("E.rate", "I.rate", "E_to_I.conductance", "I_to_E.conductance")
        ]
    ).T[indices]
    expected = np.array(reference["trajectory"]["state"])
    # Tiny numerical phase differences accumulate over the oscillatory hold.
    np.testing.assert_array_less(
        np.max(np.abs(actual - expected), axis=0), [0.02, 0.02, 1e-5, 5e-5]
    )
    for name in (
        "network.png",
        "analysis.png",
        "analysis.svg",
        "analysis.json",
        "trajectory.npz",
        "model.json",
    ):
        assert (tmp_path / name).stat().st_size > 0
