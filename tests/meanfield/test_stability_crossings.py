"""Analytic systems check root locations, scan direction and bracket handling."""

import numpy as np
import pytest

from snnlab import meanfield as mf
from snnlab.meanfield import analysis, closures


def linear_model(kind="crossing", omega=0.2):
    ctx = closures.BuildContext({}, {}, {}, {}, {}, "dynamic", "reject")
    x = ctx.state("x", unit="1")
    y = ctx.state("y", unit="1")
    control = ctx.parameter("control", 0, unit="1")
    offset = ctx.parameter("offset", 0, unit="1")
    tau = ctx.parameter("tau", 1, unit="ms")
    if kind == "crossing":
        growth = (control - offset) / tau
    elif kind == "tangent":
        growth = -((control - offset) ** 2) / tau
    elif kind == "multiple":
        growth = (control - 0.5) * (control + 0.5) / tau
    else:
        growth = 0
    ctx.derivative(x, growth * x - omega * y)
    ctx.derivative(y, omega * x + growth * y)
    return ctx.build()


def branch(model, values):
    return mf.continuation(model, "control", values, guess={"x": 0, "y": 0})


@pytest.mark.parametrize(
    "values,before,after", [([-1, 1], True, False), ([1, -1], False, True)]
)
def test_refines_oscillatory_crossing_in_scan_order(values, before, after):
    model = linear_model()
    (crossing,) = mf.find_stability_crossings(model, "control", branch(model, values))
    assert isinstance(crossing, mf.StabilityCrossing)
    assert crossing.parameter == "control"
    assert crossing.value == pytest.approx(0, abs=1e-10)
    assert crossing.bracket == tuple(values)
    assert crossing.stable_before is before and crossing.stable_after is after
    assert crossing.oscillatory
    assert crossing.frequency_hz == pytest.approx(1000 * 0.2 / (2 * np.pi))
    assert crossing.linearisation.max_real == pytest.approx(0, abs=1e-10)
    assert crossing.equilibrium.parameters["control"] == crossing.value
    np.testing.assert_allclose(list(crossing.equilibrium.state.values()), [0, 0])


def test_refinement_preserves_the_branchs_other_parameter_values():
    model = linear_model()
    changed = mf.substitute(model, {"offset": 0.37})
    (crossing,) = mf.find_stability_crossings(
        model, "control", branch(changed, [-1, 1])
    )
    assert crossing.value == pytest.approx(0.37, abs=1e-10)
    assert crossing.equilibrium.parameters["offset"] == 0.37


@pytest.mark.parametrize("values", [[-1, 0, 1], [1, 0, -1]])
def test_exact_grid_zero_is_reported_once(values):
    model = linear_model()
    crossings = mf.find_stability_crossings(model, "control", branch(model, values))
    assert len(crossings) == 1 and crossings[0].value == 0


@pytest.mark.parametrize(
    "values,before,after", [([0, 1], None, False), ([-1, 0], True, None)]
)
def test_endpoint_zero_has_an_unknown_side(values, before, after):
    model = linear_model()
    (crossing,) = mf.find_stability_crossings(model, "control", branch(model, values))
    assert crossing.value == 0
    assert crossing.stable_before is before and crossing.stable_after is after


def test_multiple_crossings_keep_branch_order():
    model = linear_model("multiple")
    result = mf.find_stability_crossings(
        model, "control", branch(model, [1, 0.2, -0.2, -1])
    )
    np.testing.assert_allclose([c.value for c in result], [0.5, -0.5], atol=1e-10)
    assert result[0].stable_before is False and result[0].stable_after is True
    assert result[1].stable_before is True and result[1].stable_after is False


def test_real_crossing_is_not_classified_as_oscillatory():
    model = linear_model(omega=0)
    (crossing,) = mf.find_stability_crossings(model, "control", branch(model, [-1, 1]))
    assert not crossing.oscillatory and crossing.frequency_hz == 0


def test_no_crossing_and_tangency_return_no_records():
    model = linear_model()
    assert mf.find_stability_crossings(model, "control", ()) == ()
    assert mf.find_stability_crossings(model, "control", branch(model, [-2, -1])) == ()
    tangent = linear_model("tangent")
    assert (
        mf.find_stability_crossings(tangent, "control", branch(tangent, [-1, 0, 1]))
        == ()
    )


@pytest.mark.parametrize("values", [[-1, -1], [-1, 1, 0]])
def test_rejects_duplicate_or_nonmonotonic_samples(values):
    model = linear_model()
    with pytest.raises(ValueError, match="strictly monotonic"):
        mf.find_stability_crossings(model, "control", branch(model, values))


def test_adjacent_grid_zeros_are_ambiguous():
    model = linear_model("flat")
    with pytest.raises(ValueError, match="isolated crossing"):
        mf.find_stability_crossings(model, "control", branch(model, [-1, 1]))


def test_rejects_changes_to_other_parameters():
    model = linear_model()
    changed = mf.substitute(model, {"offset": 0.3})
    mixed = (branch(model, [-1])[0], branch(changed, [1])[0])
    with pytest.raises(ValueError, match="vary only"):
        mf.find_stability_crossings(model, "control", mixed)


@pytest.mark.parametrize("tolerance", [0, -1, float("nan")])
def test_invalid_tolerance(tolerance):
    with pytest.raises(ValueError):
        mf.find_stability_crossings(linear_model(), "control", (), tolerance=tolerance)


def test_refinement_failure_reports_the_parameter_and_bracket(monkeypatch):
    model = linear_model()
    samples = branch(model, [-1, 1])

    def fail(*args, **kwargs):
        raise RuntimeError("equilibrium solve failed")

    monkeypatch.setattr(analysis, "find_equilibrium", fail)
    with pytest.raises(RuntimeError, match=r"control in \(-1.0, 1.0\)"):
        mf.find_stability_crossings(model, "control", samples)
