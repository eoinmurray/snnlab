"""Numerical operations compiled from the model's symbolic equations."""

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, least_squares

from snnlab.meanfield.model import magnitude


@dataclass(frozen=True)
class Equilibrium:
    state: object
    parameters: object
    residual: float


@dataclass(frozen=True)
class Linearisation:
    jacobian: np.ndarray
    eigenvalues: np.ndarray

    @property
    def max_real(self):
        return float(self.eigenvalues.real.max())


@dataclass(frozen=True)
class StabilityCrossing:
    """Refined leading-eigenvalue crossing; before/after follow branch scan order."""

    parameter: str
    value: float
    equilibrium: Equilibrium
    linearisation: Linearisation
    bracket: tuple[float, float]
    stable_before: bool | None
    stable_after: bool | None
    oscillatory: bool

    @property
    def eigenvalues(self):
        return self.linearisation.eigenvalues

    @property
    def frequency_hz(self):
        critical = self.eigenvalues[np.argmax(self.eigenvalues.real)]
        return (
            float(abs(critical.imag) * 1000 / (2 * np.pi)) if self.oscillatory else 0.0
        )


@dataclass(frozen=True)
class Trajectory:
    time_ms: np.ndarray
    state: object


@dataclass(frozen=True)
class PiecewiseLinear:
    """Scalar parameter schedule; times in ms, endpoint values held outside."""

    times: object
    values: object

    def __post_init__(self):
        times = tuple(magnitude(t, "ms") for t in self.times)
        values = tuple(float(v) for v in self.values)
        if (
            len(times) < 2
            or len(times) != len(values)
            or np.any(np.diff(times) <= 0)
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                "schedule needs finite values and strictly increasing times"
            )
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "values", values)

    def __call__(self, time):
        return float(np.interp(time, self.times, self.values))


def evaluate(model, state, *, parameters=None):
    vector = model.vector(state)
    result = np.asarray(
        model.numerical_rhs(*vector, *model.values(parameters)), dtype=float
    )
    if not np.isfinite(result).all():
        raise ValueError("model derivative is nonfinite")
    return result


def find_equilibrium(model, *, guess, parameters=None, tolerance=1e-9):
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    vector = model.vector(guess)
    values = model.values(parameters)
    bounds = np.array(
        [model.bounds.get(name, (-np.inf, np.inf)) for name in model.states]
    ).T
    scale = np.array(
        [0.01 if model.units[name] == "uS" else 1.0 for name in model.states]
    )

    def residual(state):
        return np.asarray(model.numerical_rhs(*state, *values)) / scale

    def jac(state):
        return np.asarray(model.numerical_jacobian(*state, *values)) / scale[:, None]

    solution = least_squares(
        residual,
        vector,
        jac=jac,
        bounds=bounds,
        x_scale=scale,
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
        max_nfev=2000,
    )
    error = float(np.max(np.abs(residual(solution.x))))
    if not solution.success or not np.isfinite(error) or error > tolerance:
        raise RuntimeError(
            f"equilibrium not found; scaled derivative residual={error:g}"
        )
    return Equilibrium(
        MappingProxyType(dict(zip(model.states, solution.x))),
        MappingProxyType(dict(zip(model.parameters, values))),
        error,
    )


def linearise(model, *, at, parameters=None):
    if isinstance(at, Equilibrium):
        if parameters is None:
            parameters = at.parameters
        at = at.state
    vector = model.vector(at)
    matrix = np.asarray(
        model.numerical_jacobian(*vector, *model.values(parameters)), dtype=float
    )
    return Linearisation(matrix, np.linalg.eigvals(matrix))


def integrate(
    model,
    *,
    duration,
    initial,
    parameters=None,
    controls=None,
    sample_interval=0.5,
    method="LSODA",
    rtol=1e-8,
    atol=1e-10,
    max_step=0.5,
):
    duration = magnitude(duration, "ms")
    interval = magnitude(sample_interval, "ms")
    step = magnitude(max_step, "ms")
    if min(duration, interval, step) <= 0:
        raise ValueError("duration, sample interval and maximum step must be positive")
    vector = model.vector(initial)
    baseline = dict(zip(model.parameters, model.values(parameters)))
    controls = dict(controls or {})
    if set(controls) - set(model.parameters):
        raise ValueError("controls reference an unknown parameter")
    # A moving proportionality factor would need an additional derivative term.
    protected = set().union(
        *(
            expr.free_symbols - set(model.states.values())
            for expr in model.reconstructions.values()
        )
    )
    protected |= set().union(*(expr.free_symbols for expr in model.constraints))
    if any(model.parameters[name] in protected for name in controls):
        raise ValueError(
            "controls cannot vary matched-decay constraints or reconstruction ratios"
        )
    indices = {name: i for i, name in enumerate(model.parameters)}
    baseline_values = tuple(baseline.values())

    def rhs(time, state):
        values = list(baseline_values)
        for name, control in controls.items():
            value = magnitude(control(time), model.units[name])
            lower, upper = model.parameter_bounds.get(name, (-np.inf, np.inf))
            if not lower <= value <= upper:
                raise ValueError(f"invalid controlled parameter: {name}")
            values[indices[name]] = value
        return model.numerical_rhs(*state, *values)

    times = np.arange(0, duration, interval)
    times = np.append(times, duration)
    result = solve_ivp(
        rhs,
        (0, duration),
        vector,
        t_eval=times,
        method=method,
        rtol=rtol,
        atol=atol,
        max_step=step,
    )
    if not result.success or not np.isfinite(result.y).all():
        raise RuntimeError(f"integration failed: {result.message}")
    return Trajectory(result.t, MappingProxyType(dict(zip(model.states, result.y))))


def continuation(model, parameter, values, *, guess):
    """Follow one equilibrium branch; this does not enumerate all equilibria."""
    rows = []
    for value in values:
        equilibrium = find_equilibrium(
            model, guess=guess, parameters={parameter: value}
        )
        stability = linearise(model, at=equilibrium)
        rows.append((float(value), equilibrium, stability))
        guess = equilibrium.state
    return tuple(rows)


def find_stability_crossings(
    model,
    parameter,
    branch,
    *,
    tolerance=1e-10,
    eigenvalue_tolerance=1e-8,
):
    """Refine sign changes of the largest eigenvalue real part along a branch.

    Uses Brent's method with fresh equilibrium and Jacobian evaluations, holding
    the branch's other parameters fixed. Values must be strictly monotonic.
    Exact grid zeros are reported once when signs differ across them; endpoint
    zeros have an unknown stability on the missing side. Tangencies are skipped,
    and adjacent exact zeros are rejected as ambiguous. This does not certify
    a Hopf bifurcation or reliably track branches through folds.
    """
    if parameter not in model.parameters:
        raise ValueError(f"unknown parameter: {parameter}")
    tolerance = magnitude(tolerance, model.units[parameter])
    if (
        tolerance <= 0
        or not np.isfinite(eigenvalue_tolerance)
        or eigenvalue_tolerance < 0
    ):
        raise ValueError(
            "tolerance must be positive and eigenvalue_tolerance nonnegative"
        )
    rows = tuple(branch)
    if not rows:
        return ()
    values = np.asarray([row[0] for row in rows], dtype=float)
    if not np.isfinite(values).all() or (
        len(rows) > 1
        and not (np.all(np.diff(values) > 0) or np.all(np.diff(values) < 0))
    ):
        raise ValueError(
            "branch parameter values must be finite and strictly monotonic"
        )
    baseline = dict(rows[0][1].parameters)
    cache = {}
    leading = []
    for value, equilibrium, stability in rows:
        if set(equilibrium.parameters) != set(model.parameters):
            raise ValueError("branch parameters do not match the model")
        if equilibrium.parameters[parameter] != value or any(
            equilibrium.parameters[name] != baseline[name]
            for name in baseline
            if name != parameter
        ):
            raise ValueError("branch must vary only the requested parameter")
        model.vector(equilibrium.state)
        model.values(equilibrium.parameters)
        if (
            stability.jacobian.shape != (len(model.states), len(model.states))
            or stability.eigenvalues.shape != (len(model.states),)
            or not np.isfinite(stability.eigenvalues).all()
        ):
            raise ValueError("branch linearisation does not match the model")
        cache[float(value)] = (equilibrium, stability)
        leading.append(stability.max_real)
    if any(left == right == 0 for left, right in zip(leading[:-1], leading[1:])):
        raise ValueError(
            "adjacent zero-stability points do not identify an isolated crossing"
        )

    def evaluate_at(value, guess):
        value = float(value)
        if value not in cache:
            equilibrium = find_equilibrium(
                model, guess=guess, parameters={**baseline, parameter: value}
            )
            cache[value] = (equilibrium, linearise(model, at=equilibrium))
        return cache[value]

    candidates = []
    for index, (left, right) in enumerate(zip(rows[:-1], rows[1:])):
        if not (
            (leading[index] < 0 < leading[index + 1])
            or (leading[index] > 0 > leading[index + 1])
        ):
            continue
        try:
            value = brentq(
                lambda value: evaluate_at(value, left[1].state)[1].max_real,
                min(left[0], right[0]),
                max(left[0], right[0]),
                xtol=tolerance,
            )
        except (ValueError, RuntimeError) as error:
            raise RuntimeError(
                f"stability crossing refinement failed for {parameter} in ({left[0]}, {right[0]}): {error}"
            ) from error
        evaluate_at(value, left[1].state)
        candidates.append(
            (
                index + 0.5,
                value,
                (left[0], right[0]),
                leading[index] < 0,
                leading[index + 1] < 0,
            )
        )
    if len(rows) > 1:
        for index, real in enumerate(leading):
            if real != 0:
                continue
            before = leading[index - 1] < 0 if index else None
            after = leading[index + 1] < 0 if index + 1 < len(rows) else None
            if before is not None and after is not None and before == after:
                continue
            candidates.append(
                (
                    index,
                    float(values[index]),
                    (
                        float(values[max(0, index - 1)]),
                        float(values[min(len(rows) - 1, index + 1)]),
                    ),
                    before,
                    after,
                )
            )
    crossings = []
    for _, value, bracket, before, after in sorted(
        candidates, key=lambda item: item[0]
    ):
        equilibrium, stability = cache[float(value)]
        critical = stability.eigenvalues[np.argmax(stability.eigenvalues.real)]
        crossings.append(
            StabilityCrossing(
                parameter,
                float(value),
                equilibrium,
                stability,
                tuple(map(float, bracket)),
                before,
                after,
                bool(abs(critical.imag) > eigenvalue_tolerance),
            )
        )
    return tuple(crossings)
