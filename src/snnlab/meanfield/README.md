# Qualitative mean-field modelling

`snnlab.meanfield` constructs symbolic population models from a `lang.Network`
or compiled `lang.Bundle`. Manipulation, symbolic inspection and numerical
analysis are available in the main namespace. Scientific approximation families
and their public extension API live in `snnlab.meanfield.closures`.

These models explore feedback, timescales and qualitative dynamical transitions.
The reduction does not assert that spiking-network trajectories, rates or
frequencies will agree quantitatively. Comparisons with an existing mean-field
implementation check equation reproduction, rather than spiking-model accuracy.

## Build and inspect

```python
from snnlab import lang, meanfield as mf
from snnlab.meanfield import closures

model = mf.reduce(
    bundle,
    closure=closures.RateConductance(
        gain=closures.NoisyLIF(sigma=4 * lang.mV),
        tau_rate={"E": 20 * lang.ms, "I": 5 * lang.ms},
        driving_force="resting",
    ),
    weights=mf.MeanIncomingWeights(
        incoming_weight_sums,
        provenance="effective sampled matrices, summed over sources and averaged over targets",
    ),
    inputs={"X": mf.StationaryRate(2.5 * lang.Hz)},
    external_filters="quasistatic",
    delays="omit",
)

print(mf.summary(model))
equations = mf.equations(model)  # mapping: state name -> SymPy Eq
matrix = mf.jacobian(model)  # symbolic Jacobian
tex = mf.latex(model)  # aligned equations, for embedding in a document
```

1. The initial built-in closure supports spiking `COBA_LIF` populations and
   exponential `AMPA`/`GABA` pathways. It uses homogeneous source rates, stationary
   noisy-LIF gains and fixed resting driving forces. Synaptic decay and membrane
   parameters are read from the graph. `NoisyLIF` currently requires positive
   refractory durations and a fixed positive voltage-noise scale.
2. `MeanIncomingWeights` accepts projection IDs mapped to
   `sum(effective_weight_matrix) / number_of_targets`, in uS. This includes zero
   edges, excluded autapses, constraints and simulator weight scaling. No
   initializer expectation or weight sampling is performed.
3. Alternatively, `RealisedWeights` accepts parameter IDs mapped to effective
   dense matrices. The default layout is the simulator's native
   `(sources, targets)`; use `layout="target_source"` for `(targets, sources)`.
   These must already have simulator preprocessing applied. Compact spatial
   kernels and raw initializer tensors are not effective dense matrices.
4. `StationaryRate` specifies only a mean rate, in Hz. It does not imply Poisson
   statistics. Shared-source covariance is omitted by this closure.
5. `external_filters="dynamic"` is the default and retains input filters.
   `"quasistatic"` substitutes the stationary mean even during gain changes,
   intentionally omitting filtering lag, as exp148 does.
6. Nonzero graph delays fail by default; `delays="explicit"` retains transmission
   delays and `delays="omit"` explicitly makes an ODE approximation. Select
   `modulation="translate"` for supported graph schedules or supply explicit controls.
7. Built-in names include `E.rate`, `E.tau_rate`, `E.applied_current`,
   `E_to_I.conductance`, `E_to_I.coupling`, `E_to_I.tau` and `E_to_I.gain`.
   State/parameter mappings and declared units are available on `model`.

`latex` uses mathematical population labels such as `r_E`, `g_e^I` and `I_E`,
with exact auxiliary definitions for currents, external conductances and reduced
state reconstructions. `mf.latex(model, expanded=True)` inlines those expressions.
`mf.equations(model)` continues to return the computational SymPy equations.
Custom closures can set `ctx.notation[symbol]` to a LaTeX label and use
`ctx.display_definition(name, expression, latex=label)` to name an expression
for presentation. Display metadata does not change numerical dynamics.

## Manipulate and analyse

```python
reduced = mf.reduce_states(
    model,
    method="matched_decay",
    retain=["E_to_I.conductance", "I_to_E.conductance"],
)
changed = mf.substitute(reduced, {"X_to_I.gain": 0.5})

equilibrium = mf.find_equilibrium(
    changed,
    guess={
        "E.rate": 2,
        "I.rate": 6,
        "E_to_I.conductance": 0.01,
        "I_to_E.conductance": 0.05,
    },
)
stability = mf.linearise(changed, at=equilibrium)
trajectory = mf.integrate(
    reduced,
    duration=2300 * lang.ms,
    initial=equilibrium.state,
    controls={
        "X_to_I.gain": mf.PiecewiseLinear(
            times=[800 * lang.ms, 1550 * lang.ms],
            values=[1, 0],
        )
    },
)
branch = mf.continuation(
    reduced,
    "X_to_I.gain",
    [1, 0.75, 0.5, 0.25, 0],
    guess=equilibrium.state,
)
```

Transformations return new models. `substitute` changes numerical defaults while
retaining symbolic parameter identities. `reduce_states` merges outgoing
recurrent filters with matched decay times. It records the expressions that
removed initial conductances must satisfy in `model.reconstructions`; arbitrary
initial conductances require the unreduced model. The equality constraints are
checked again when parameter values change. Controls that move reconstruction
ratios or matched-decay constraints are rejected.

`find_equilibrium` returns named states, the parameter values used and the maximum
scaled derivative residual. Its residual scales conductance derivatives by
0.01 uS and other derivatives by one native state unit. The default tolerance
is `1e-9` in those scaled derivative units. `linearise` automatically reuses the
parameters stored with an equilibrium unless explicitly overridden.
`continuation` follows a single branch with the preceding equilibrium as its
next guess; it does not enumerate multiple equilibria or classify a Hopf
bifurcation. Eigenvalues describe local stability of the chosen mean-field ODE.

Time is in ms, rates in Hz, conductances in uS and currents in nA. Bare numerical
arguments use these conventions; quantities with incompatible declared units
are rejected. Bounds constrain equilibrium searches. Custom trajectories are
not automatically projected onto physical state bounds.

## Refine stability crossings

```python
branch = mf.continuation(
    model,
    "X_to_I.gain",
    gains,
    guess=guess,
)
crossings = mf.find_stability_crossings(
    model,
    "X_to_I.gain",
    branch,
    tolerance=1e-10,
)
for crossing in crossings:
    print(crossing.value, crossing.frequency_hz)
```

The helper brackets sign changes in the largest eigenvalue real part and
refines each with Brent's method. It recomputes equilibria and linearisations at
trial values while preserving the branch's other parameter values. Input values
must be strictly ascending or descending.

Each `StabilityCrossing` contains `parameter`, `value`, `equilibrium`,
`linearisation`, `eigenvalues`, `bracket`, `stable_before`, `stable_after`,
`oscillatory` and `frequency_hz`. Before/after follow the supplied scan order,
including descending scans. Stability is unknown (`None`) on the missing side
of an exact endpoint crossing. Real crossings have zero oscillatory frequency.

`tolerance` is the absolute root-location tolerance in the varied parameter's
native units. `eigenvalue_tolerance` defaults to `1e-8` radians/ms and determines
whether the leading mode's imaginary part is classified as oscillatory.
Exact grid zeros are reported once when signs change across them; tangencies
are skipped and adjacent exact zeros are rejected as ambiguous. Refinement
failures raise an error identifying the parameter and bracket.

This finds sampled stability-crossing candidates. It does not establish Hopf
criticality, detect every crossing between coarse samples, or reliably track
an equilibrium branch through folds.

## Export a model document

```python
import json

path = model.write("results/model.json")
saved = json.loads(path.read_text())
rate_equation = saved["equations"]["E.rate"]["latex"]
current_equation = saved["equations"]["E.effective_current"]["latex"]
```

`write` atomically replaces one JSON file and returns its `Path`. The
`snnlab.meanfield.model/v1` document contains:

1. `states`, `parameters`, `state_order` and `parameter_order`: symbols, display
   notation, units, bounds and numerical parameter values. An unbounded endpoint
   is JSON `null`.
2. `model_definition`: computational and display expressions, auxiliary
   definitions, reconstruction constraints, filter topology and function
   specifications. Expressions use the declared `sympy-str` format.
3. `equations`: stable IDs mapped to equation kind, individually renderable
   `latex`, `lhs_latex`, `rhs_latex` and direct symbol `dependencies`.
   `equation_order` preserves the generated presentation order.
4. `assumptions`, `provenance` and `summary`: the approximation choices and their
   source metadata, plus a readable description.
5. `symbols`: a glossary of states, parameters, derived quantities and functions.
   It retains their meanings and, for shorthands, `definition_latex` and the
   relevant component symbols. Coupling entries record
   `C = sum(effective weights) / target_count`, target/source sizes, weight
   units, index meanings and zero-edge handling. The same coupling metadata
   appears in `parameters`.
6. `latex`: the combined block for callers that want the complete presentation.

State IDs such as `E.rate` identify differential equations. Definition IDs such
as `E.effective_current` identify auxiliary relations. Equation selectors can
choose either kind without splitting or parsing a LaTeX block. For example,
`saved["symbols"]["E_to_I.coupling"]["definition_latex"]` retrieves the coupling
normalisation formula. Custom closures can use
`ctx.describe(symbol, description="...", definition_latex="...", ...)` to
record the semantics of their own shorthand symbols.

The document describes the symbolic model and numerical defaults. Python
callbacks remain runtime implementations; custom functions can supply an
inspectable JSON specification through `ctx.function(..., definition={...})`.
The built-in noisy-LIF gain includes its membrane, noise and quadrature settings.
No callback code is executed when writing the document.

## Extend closures

Implement the public `closures.Closure` protocol: one `build(ctx)` method
returning an `mf.Model`. No registration or subclass is required. Built-ins use
the same `closures.BuildContext` construction methods:

```python
class LinearRate:
    def build(self, ctx: closures.BuildContext) -> mf.Model:
        for population in ctx.populations:
            name = population["id"]
            rate = ctx.state(f"{name}.rate", unit="Hz", bounds=(0, float("inf")))
            target = ctx.parameter(f"{name}.target", 10, unit="Hz")
            tau = ctx.parameter(f"{name}.tau", 5 * lang.ms, unit="ms")
            ctx.derivative(rate, (target - rate) / tau)
        ctx.assumptions.append("Independent linear rate relaxation; no coupling")
        return ctx.build()
```

Use that illustrative closure with a network without projections and
`weights=mf.MeanIncomingWeights({}, provenance="no projections")`. A coupled
custom closure can traverse `ctx.projections`, `ctx.weights`, `ctx.inputs` and
`ctx.applied_current` to assemble its own equations. It is responsible for
handling or rejecting the graph components it consumes.

`ctx.function(name, callable, derivative=callable)` binds a scalar symbolic
function and its analytic first derivative. Numerical integration and symbolic
Jacobian compilation then work through the same pipeline. Higher derivatives
of these opaque functions require additional symbolic definitions.
An interchangeable gain implements `bind(ctx, population)` returning that
symbolic callable and a maximum rate in Hz, as `NoisyLIF` does.

Units are preserved as metadata and quantity boundaries are checked; arbitrary
custom SymPy expressions do not yet receive automatic dimensional analysis.

## Explicit transmission delays

Choose `delays="explicit"` in `reduce` to retain fixed nonnegative graph delays
per projection. Zero delay uses the current source rate. `delays="reject"`
remains the default; `delays="omit"` records an explicit ODE approximation.
Dynamic AMPA/GABA filters implement

$$
\frac{d g_p(t)}{dt}=-\frac{g_p(t)}{\tau_p}
+\frac{C_p a_p(t) r_{s(p)}(t-d_p)}{1000}.
$$

Here $p$ is a projection, $s(p)$ its source population or external input,
$t$ is time in ms, $g_p$ is mean conductance in µS, $\tau_p$ its decay time in ms,
$C_p$ is mean incoming effective weight in µS, $a_p$ is dimensionless gain,
$r_{s(p)}$ is source rate in Hz and $d_p$ is transmission delay in ms. The factor
1000 converts Hz to inverse ms. This is the existing rate-conductance closure
with explicit delayed forcing, not a new closure or a filter-chain approximation.

```python
trajectory = mf.integrate(
    model,
    duration=600,
    initial=initial,
    history=mf.ConstantHistory(initial),
    controls={
        "E_A_to_E_B.gain": mf.PiecewiseConstant(
            times=[500],
            values=[0, 1],
        ),
    },
    method="DOP853",
    rtol=1e-9,
    atol=1e-11,
    max_step=0.1,
    sample_interval=0.5,
)
trajectory.write("results/trajectory.json")
```

### History and units

A nonzero delayed dependency requires an explicit `ConstantHistory(state)` or
`History(function, start, description, breakpoints=())`. The callback takes time
in ms and returns **every retained state by name**, in its declared unit; bare
numbers use native units. The required interval is $[-d_{\max},0]$, where
$d_{\max}$ is the largest retained delay in ms. `History.start` must cover this
interval. Supply a continuous, piecewise smooth history and declare its interior
corners through `breakpoints`. `description` records the mathematical specification
in exports. For example:

```python
history = mf.History(
    function=lambda t: {**initial, "E_A.rate": initial["E_A.rate"] + 0.001 * t},
    start=-2,
    description="E_A rate varies linearly at 0.001 Hz/ms; other states constant",
)
```

History at zero must agree with `initial` to absolute tolerance `1e-12` in each
native state unit. Names, shape, finiteness and quantity units are validated.
The solver probes histories initially and validates every later lookup; it cannot
prove an arbitrary callback continuous. Initial derivative agreement is not
required: the initial derivative corner is propagated through the delays.
External source rates remain stationary parameters unless controlled; a controlled
external rate is evaluated at emission time `t-delay`, including negative times.
Such callbacks must define their pre-zero values. Nonzero explicit external delays
require `external_filters="dynamic"`; quasistatic substitution is rejected.

### Solver, interpolation and switches

The solver uses an adaptive **method of steps**: each interval is no longer than
the smallest positive delay, so delayed states come from already completed
intervals. It retains SciPy `solve_ivp` dense output, using the selected method's
interpolant: RK23 cubic Hermite, RK45 quartic, DOP853 seventh-order, or LSODA's
variable-order polynomial. `rtol` and `atol` control local ODE error on each
interval, not a certified global DDE error. `atol` may be one native-unit value
per state. `max_step` limits internal steps and therefore interpolation spacing;
`sample_interval` only selects observations and may exceed a delay. Establish
convergence by tightening tolerances and reducing `max_step`.
See [SciPy solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)
for solver-specific interpolation and tolerance semantics, and
[Shampine and Thompson (2001)](https://doi.org/10.1016/S0168-9274(00)00010-8)
for numerical delay-equation methods and discontinuity tracking.

`PiecewiseConstant(times, values)` has one more value than switch times and is
right-continuous. `PiecewiseLinear` exposes its knot times too. Both trigger
split/restart with continuous dynamical state; no solver step straddles a known
switch. For arbitrary callbacks, supply all corners/jumps explicitly through
`integrate(breakpoints=[...])` or a control's `breakpoints` attribute. Smooth
callbacks need no breakpoints. Initial/history/control corners are propagated
through all delay sums. Delay values and seed times are quantized once to
integer ticks of `1e-12` ms for restart scheduling; propagation and deduplication
use integer addition, so equivalent paths cannot drift apart on long runs.
The derivative still uses the authored delays. More than 100,000 distinct
boundaries raises a complexity error rather than
silently skipping them. Very small or incommensurate delays can be expensive.
Retained dense history uses memory proportional to solver steps over the run.

Modulation acts at **target delivery**: `gain(t) * source_rate(t-delay)`, matching
the graph executor. Changing gain does not erase conductance already present.
`modulation="translate"` in `reduce` explicitly translates graph constant, step,
piecewise-constant and ramp gains into `model.controls`, which `integrate` uses
by default. Explicit `controls` override matching entries. Step/piecewise switch
times are rounded to the first graph timestep at or after the authored time,
matching graph sampling. Ramps are continuous interpolations agreeing at graph
sample times. Other graph modulation kinds fail clearly; use an explicit
control instead. No graph schedule is silently discarded.

### Analysis and exported evidence

For delayed models, `find_equilibrium` solves the constant-history stationary
balance, replacing delayed rates by their constant values. It freezes parameter
defaults/overrides and ignores time schedules. This is mathematically valid for
an autonomous frozen system and makes **no stability claim**. `evaluate`, which
has no history argument, rejects delayed derivatives. `jacobian`, `linearise`,
`continuation` and `find_stability_crossings` reject delayed stability analysis;
characteristic roots are not implemented. `reduce_states` conservatively rejects
all nonzero delayed models: matching decay times alone does not establish
proportional histories. `substitute` remains available for non-delay parameters.

`equations` and LaTeX expose source rates at `t-delay`. Model JSON includes each
authored projection delay, selected delay mode, symbolic delayed dependencies,
required history interval and default controls. `Trajectory.metadata` and
`Trajectory.write(path)` additionally record actual history, initial values,
parameters, controls, solver settings and interpolation. Runtime callback code
is not serialized; record its mathematical definition and preserve its source.
The exported `delay_approximation="none"` describes transmission only: gain,
noise and population-homogeneity assumptions remain.

The runnable `examples/meanfield-delays/example.py` compares two coupled E/I
circuits with an uncoupled run and a 500 ms coupling onset. It retains 0.1 ms
internal and 2 ms cross-circuit delays. It does not claim sustained gamma,
exp133 reproduction, locking boundaries or delayed-system bifurcations.

## Scientific sources

1. The [Pinglab mean-field note](https://github.com/eoinmurray/pinglab/blob/main/md/notes/mean-field-model.mdx)
   describes the rate-conductance closure, fixed driving-force approximation and
   matched-decay reduction used in the first example.
2. Kreutz-Delgado, [Mean Time-to-Fire for the Noisy LIF Neuron](https://doi.org/10.48550/arXiv.1501.04032),
   derives the stationary Siegert firing-rate formula. The prescribed noise and
   first-order rate relaxation are additional closure assumptions.
3. See `examples/meanfield/README.md` for the exp148 reproduction and its frozen
   evidence provenance.
