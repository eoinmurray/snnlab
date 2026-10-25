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
6. Nonzero graph delays fail by default; `delays="omit"` explicitly makes an ODE
   approximation. Graph modulation schedules also fail; translate them into
   explicit numerical controls rather than silently discarding them.
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

## Scientific sources

1. The [Pinglab mean-field note](https://github.com/eoinmurray/pinglab/blob/main/md/notes/mean-field-model.mdx)
   describes the rate-conductance closure, fixed driving-force approximation and
   matched-decay reduction used in the first example.
2. Kreutz-Delgado, [Mean Time-to-Fire for the Noisy LIF Neuron](https://doi.org/10.48550/arXiv.1501.04032),
   derives the stationary Siegert firing-rate formula. The prescribed noise and
   first-order rate relaxation are additional closure assumptions.
3. See `examples/meanfield/README.md` for the exp148 reproduction and its frozen
   evidence provenance.
