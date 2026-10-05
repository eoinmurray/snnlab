# snnlang

`snnlang` is a standalone Python authoring library for graph-shaped spiking
circuits. It validates a network and optional standard training recipe, then
writes a deterministic, data-only bundle. It does not simulate or train.

```python
from snnlab import lang as snn

net = snn.Network("small_ping")
events = net.input(
    "events", shape=("time", "batch", 128), signal_type="spikes", unit="spike"
)
cell = snn.components.ping(net, name="cell", n_e=256, n_i=64, source=events)
scores = snn.readouts.MeanVoltage(
    source=cell.E.spikes, classes=10, name="classifier"
)
net.output("scores", scores)

bundle = snn.compile(net)
bundle.write("small_ping.bundle", visualise=True)
```

The bundle contains canonical `graph.json`, optional `training.json`, a
digest-bearing `manifest.json`, copied logical assets, a text summary, and
optional circuit/training/expanded SVG and PNG reports. Physical dataset and
checkpoint paths deliberately do not belong in the graph.

Conductance projection weights use `uS` (microsiemens), matching the graph
executor and its `leak_us` neuron parameters. Current projection weights use
`nA` (nanoamperes) with `CUBA_LIF`/`LIF` and `ExponentialCurrent`. Authoring and
execution validate the synapse/neuron family and parameter units; values are
never silently converted between physical units. Bundles produced before SNNLang 0.2.0 may carry an
incorrect legacy `nS` label even though their values were executed as `uS`.

`snnlang` owns the semantic projection of a bundle through `snn.diagram(...)`;
the resulting renderer-neutral diagram is rendered by `tools.snnviz`. The
existing `Bundle.visualise(...)` convenience method delegates through this same
boundary.

Inputs are graph contracts, not stimulus recipes. A time-varying spike input
uses the canonical `(time, batch, channels)` axis order and declares its signal
type and unit. Dataset selection, Poisson rates, encoders, seeds, durations, and
realised spike tensors belong to the experiment protocol. `tools/snnsim` may
generate a standard stimulus from CLI parameters or consume an exact replay
with `--input-file` or `--event-file`; the replay is optional evidence, not part
of the graph.

Dense replay is resolved before graph execution. NPY files bind to the sole
graph input; NPZ arrays bind by input id. The resolver requires exact input
coverage, matching time and batch axes, declared feature shapes, finite values,
binary spikes, and boolean or zero/one masks. It records a versioned execution
protocol containing source-file digests, resolved shapes and dtypes, dataset
identity and split when supplied, sample cap, batch size, shuffle behavior,
duration, masks, and the execution seed.

Sparse event replay uses an NPZ coordinate contract. A single-input file stores
`steps`, `batches`, `channels`, `steps_count`, and `batch_size`; multi-input
files prefix each field with the graph input id. Coordinates are zero-based
integer simulation steps ordered by step, batch, and channel. Resolution rejects
duplicates and out-of-bounds coordinates, then materializes binary spikes for
the graph executor while retaining event counts and source identity in the
protocol. Typed requests may combine event-stream spike inputs with dense mask
or continuous inputs when every binding resolves to the same time and batch
axes.

```sh
uv run python -m snnlab.sim sim \
  --executor graph \
  --bundle small_ping.bundle \
  --input-file replay.npz \
  --input-dataset-id mnist-test-sha256-... \
  --input-split test \
  --no-input-shuffle \
  --seed 17 \
  --out-dir run/
```

The resolved contract is written under `execution_protocol` in `metrics.json`.
Generated Poisson inputs are execution protocols rather than graph structure.
`PoissonInputBinding` supports a fixed rate or a rate sampled uniformly and
independently per presentation from a categorical set. The resolver owns an
explicit seed, records both configured and realized rates, and uses the graph
timestep to materialize Bernoulli-discretized homogeneous Poisson spikes.

Portable dataset snapshots use `DatasetSnapshotBinding` and remain external
immutable NPZ files. The binding authenticates the file, selects samples by a
seeded permutation and cap, binds integer labels to an optional training target,
and applies one standard `DatasetEncoder`:

- `rate_poisson`: finite normalized `(samples, channels)` features scaled by a
  maximum rate and sampled for a physical duration;
- `prebinned_spikes`: exact binary `(time, samples, channels)` replay;
- `event_bin`: timestamped sample/channel events binned at graph `dt`, with
  binary collision counts retained in provenance.

The `tools/snnsim.dataset-snapshot-binding/v1` protocol records the source digest,
dataset identity and split, selected indices, keys, sample cap, batch size,
shuffle policy, timing, encoder parameters, labels, and execution/order/encoder
seeds. No download or dataset registry is consulted.

```sh
uv run python -m snnlab.sim train \
  --executor graph \
  --bundle deep_network.bundle \
  --dataset-file shd-train-recording.npz \
  --dataset-encoder event-bin \
  --dataset-target-id gesture \
  --input-dataset-id shd-train-sha256-... \
  --input-split train \
  --t-ms 100 \
  --max-samples 1000 \
  --input-shuffle \
  --seed 17 \
  --out-dir train-run/
```

```sh
uv run python -m snnlab.sim sim \
  --executor graph \
  --bundle small_ping.bundle \
  --poisson-protocol categorical-rate \
  --input-rates 0.5 1 5 10 25 \
  --n-batch 64 \
  --t-ms 200 \
  --seed 17 \
  --out-dir run/
```

Graph validity is checked independently of a simulator backend. Passing
`target="tools/snnsim"` adds capability diagnostics but never changes the graph.
The first additive `tools/snnsim` backend route accepts the deliberately narrow
single-layer MNIST PING subset:

```sh
uv run python -m snnlab.sim sim \
  --bundle small_ping.bundle \
  --t-ms 200 \
  --out-dir run/

uv run python -m snnlab.sim train \
  --bundle classifier.bundle \
  --max-samples 1000 \
  --batch-size 64 \
  --out-dir train-run/
```

Execution choices such as duration, seed, input mode, output directory, and
recordings remain CLI concerns. Structural flags cannot override the bundle.
For the first training subset, `training.json` owns cross-entropy, AdamW,
epochs, learning rate, and a trainable input/readout plus frozen recurrent
scope; dataset cap and batch size remain execution choices.
Parameter groups are exhaustive and non-overlapping. The compiled recipe also
contains resolved trainable/frozen parameter lists and a stable per-parameter
learning-rate map; frozen groups use zero and trainable groups require a
positive finite rate.
The standard backward contract uses an explicit fast-sigmoid surrogate and
positive per-population voltage-gradient dampening factors. Compilation records
both in `resolved_gradients`; the legacy adapter maps the supported shared
dampening case back to its established CLI settings.
Training recipes may declare a physical presentation duration independently of
graph `dt`, plus the collection's exact multi-layer spike-budget penalty. Its
stored aggregation contract is the mean over presentations and layers of each
population's mean-rate squared overshoot above a ceiling in Hz.
Before serialization, reverse reachability proves that every objective and
regularizer reaches at least one trainable parameter through enabled graph
elements. The check respects frozen groups and stop-gradient boundaries and
reports the exact reachable and trainable sets when a route is absent.
Unsupported graph structures fail with an element-level capability error;
legacy commands that omit `--bundle` retain their existing defaults and
behaviour.

## Graph-native forward execution

The opt-in graph backend is exposed through typed requests. Bundle loading is
still data-only and does not import this authoring package.

```python
import torch
from snnlab.sim.execution import DenseArrayBinding, ExecutionSpec, simulate

result = simulate(ExecutionSpec(
    kind="simulate",
    executor="graph",
    bundle="small_ping.bundle",
    input_bindings=(DenseArrayBinding("events", torch.zeros(100, 1, 128)),),
))
```

Pass dense tensors, sparse events, generated Poisson spikes or dataset snapshots
as typed binding objects through the single `ExecutionSpec.input_bindings`
sequence. Dense bindings can include explicit provenance in their `source`
mapping.

The planner lowers the complete dense topology before stepping. It supports
arbitrarily named COBA-LIF and leaky-integrator populations, independent spike
inputs, AMPA and GABA projections, feedforward/recurrent/feedback paths,
integral delay buffers, standard readout operations, and explicitly exposed diagnostics from
named populations and projections. Mean voltage, final voltage, spike count, spike rate, and
cumulative-potential readouts execute through the graph operation vocabulary.
Spike-rate readouts report spikes/s from either an explicit duration in seconds
or a `(time, batch)` valid-time mask whose duration is inferred from graph
`dt`. Zero-delay feedforward edges follow a deterministic topological order.
Recurrent and feedback spikes are causal, so zero additional delay means one
simulation step. Positive delays must be exact integer multiples of the declared
timestep. Zero-delay cycles, dimension errors, malformed masks, ambiguous
durations, polarity mismatches, and missing backend capabilities fail before
simulation.

Projections authored with `enabled=False` remain in the graph with the same
parameter shape, initializer, name, and construction position, but contribute
zero conductance at runtime. Controlled cells can therefore disable a recurrent
loop without shifting later tensor identities or random initialization draws.

Initializer specifications distinguish lower-clamped and signed normals,
uniform, constant, and zero distributions. Lower-clamped normals can apply
seeded Bernoulli or exact-fan-in initial zeroing. Build metrics expose stable
per-parameter realized statistics together with the constraint, unit, runtime
shape, and effective scaling convention. `Network.connect`, `Network.parameter`,
and `readouts.MeanVoltage` accept `initialization_scaling="direct"` or
`"fan_in_normalized"`. Projection defaults remain normalized; direct scaling
skips only fan-in division after drawing, clamping and compensated initial
zeroing. The policy is serialized on the parameter, shared connections inherit
it and conflicting requests fail. Older graphs without this field preserve
normalized projections and direct operation parameters. Checkpoint loading
copies stored values without rescaling. See the documentation Parameters
reference for examples and compatibility rules.

The typed graph API supports deterministic single-batch AdamW updates for the
validated cross-entropy and spike-budget vocabulary. `ExecutionSpec` supplies
resolved inputs and external targets; a bundle may authenticate and supply its
training recipe. Results expose per-update loss components, named gradients,
parameters, and optimizer state.

Graph training accepts named integer target arrays from NPY/NPZ files through
the same provenance boundary as inputs. With a positive recipe epoch count it
iterates the input sample axis in deterministic mini-batches, optionally using
a seed-derived permutation per epoch. The execution protocol records target
digests, dataset identity and split, sample cap, batch size, shuffle policy,
epoch count, and order seed.

Graph training checkpoints use a versioned manifest plus a digest-verified
tensor payload. They key parameters and AdamW state by stable graph id and
record graph/training digests, completed updates, execution protocol,
initializer metadata, CPU random state, exact named accelerator random states,
and the exact next epoch/batch. Manifest version 2 records `cpu`, `cuda`, or
`mps` as the random backend. CUDA captures every contiguous device generator;
MPS captures its single generator. Resume requires the same backend and exact
device topology before changing any stream. CPU-only version 1 checkpoints
remain loadable. The
trainer can save final and invocation-selected checkpoints and resume exactly after rejecting recipe,
protocol, initializer, shape, dtype, or parameter-set mismatches. An explicit
one-layer legacy parameter map fails closed when any graph parameter is
unrepresentable. Mocked topology tests cover accelerator-state serialization
and fail-closed restore dispatch, but production accelerator trajectory parity
remains a hardware gate. The legacy CLI
and bundle adapter remain the default and retain their historical numerical
contract.

`tools/snnsim/conformance.py` provides the versioned, fail-closed comparison layer
for migration evidence. It compares complete named tensor layers under an
explicit exact or numerical policy, reports coverage, shape, dtype, and error
bounds, and writes `tools/snnsim.conformance-report/v1` JSON. Canonical JSON
encoding brings topology, initializer metadata, protocols, and checkpoint
coordinates into the same report as forward outputs, gradients, parameters,
and optimizer tensors. Declared tolerance rules that match no field are errors.

The first cross-backend CPU fixtures copy one complete one-layer parameter set
through the semantic legacy map and compare E/I spikes, membrane traces,
input/AMPA/GABA conductances, and mean-voltage logits with recurrence both
isolated and active. Dynamic state and parameters match exactly; logits use a
predeclared `1e-6` absolute/relative tolerance. The fixtures established that
legacy recurrent keys are one-based (`W_ee.1`,
`W_ei.1`, `W_ie.1`, and `W_ii.1`) and that the compatible mean-voltage readout
uses the legacy 2 ms output-membrane time constant.

The bounded forward accelerator check reuses the active-recurrence case. It
runs the graph executor twice on one accelerator, then compares graph and
legacy forward state on that same device under predeclared `1e-6` absolute and
relative tolerances. It deliberately excludes training, checkpoints, datasets,
and cross-device equality:

```sh
uv run python tools/snnsim/accelerator_forward.py --device mps
# or, on a CUDA host:
uv run python tools/snnsim/accelerator_forward.py --device cuda
```

The corresponding four-update backward fixture trains all six mapped tensors
and compares the complete cross-entropy trajectory, final named surrogate
gradients, constrained parameters, and AdamW step/first-moment/second-moment
tensors under the same frozen CPU policy. A two-plus-two checkpoint resume is
bit-identical to the uninterrupted graph trajectory.
Both routes apply the non-negative parameter projection after the optimizer
step; optimizer state itself remains the unconstrained AdamW update record.

`import_legacy_parameters_v1` and `export_legacy_parameters_v1` provide
bidirectional parameter-only interchange for the supported one-layer legacy
state. They require exact keys, runtime shapes, floating dtypes, and complete
graph coverage, and return mapping-version/direction provenance. Legacy
optimizer objects are not presented as portable graph training checkpoints.

Graph simulation/inference can load a selected or final training-checkpoint
directory after authenticating its payload, graph digest, names, shapes, and
dtypes. Inference metrics retain checkpoint format/path, graph and training
digests, completed update, and selected loss; optimizer and iterator state are
not restored for inference. Non-directory checkpoints remain the explicit
legacy PyTorch state-file route.

Inference variations use the request-local `tools/snnsim.inference-overrides/v1`
contract. Generated Poisson inputs may override a positive, timestep-aligned
duration and a finite non-negative rate. Named graph projections may be scaled
by finite non-negative factors after checkpoint loading. The original graph and
checkpoint stay unchanged, and metrics record the requested and resolved
values. Unknown projection ids, mixed replay/Poisson duration requests, and
non-resampleable replay tensors fail closed. A `timestep_ms` override recompiles
an immutable graph copy, rebuilds physical decays and delays, and resamples only
generated Poisson bindings while preserving their physical duration. Runtime
state is not converted. Checkpoints authenticate against the source graph
before named, same-shaped parameters load into the effective graph.

```python
result = simulate(ExecutionSpec(
    kind="simulate",
    executor="graph",
    bundle="small_ping.bundle",
    checkpoint="train-run/selected.checkpoint",
    input_bindings=(binding,),
    options={"inference_overrides": {
        "duration_ms": 400.0,
        "input_rate_hz": 25.0,
        "projection_scales": {"cell_I_to_E": 0.5},
    }},
))
```

The CLI accepts repeatable `--scale-projection ID=FACTOR`; generated Poisson
duration and rate continue to use `--t-ms` and `--input-rate`, and
`--inference-timestep-ms` requests the recompiled timebase.

Ordered hidden-spike interventions use
`tools/snnsim.inference-interventions/v1`. `drop_spikes` removes emitted spikes by
a finite probability in `[0, 1]`; `add_poisson_spikes` unions them with a
seeded homogeneous Poisson stream at a finite non-negative rate supported by
the graph timestep. Targets are exact ids of spiking populations. Modified
spikes flow through downstream zero-delay projections, delay histories,
recordings, and readouts. Seed streams are keyed by absolute execution step, so
runtime continuation matches an uninterrupted request exactly.

```sh
uv run python -m snnlab.sim sim \
  --executor graph \
  --bundle small_ping.bundle \
  --input-file replay.npz \
  --intervention drop:cell_E=0.25 \
  --intervention add:cell_E=5 \
  --seed 17 \
  --out-dir intervention-run/
```

Every graph CLI inference directory also contains
`inference-manifest.json` using `tools/snnsim.inference-artifacts/v1`. It records
the graph and request identity, request seed, and the names, shapes, dtypes, and
SHA-256 digest of each NPZ payload. The request digest binds the execution
protocol, checkpoint, overrides, interventions, diagnostics flag, and device.
Use `validate_inference_artifacts(path, graph=graph, seed=seed)` before cache
reuse; it rejects manifest drift, a different graph or seed, missing files,
payload corruption, and array-inventory changes. Task-specific accuracy and
raster aggregation remain experiment-level operations over these stable named
tensors.

`derive_inference_products` provides the standard conversion layer for those
public tensors. Given an authenticated inference directory, an exact logits id,
integer labels, and exact population-spike recording ids, it writes versioned
labels, predictions, accuracy, per-presentation per-cell rates in Hz, and
sparse zero-based time/batch/cell rasters. The
`tools/snnsim.derived-inference/v1` manifest authenticates all derived payloads and
retains the source artifact digest. `validate_derived_inference_products`
rejects corruption or reuse against a different source cache. Scientific
acceptance thresholds remain campaign decisions rather than executor defaults.

### Diagnostic retention

`net.output` declares results returned in `result.outputs`. `net.expose` declares diagnostics returned by default in `result.diagnostics`; pass `diagnostics=False` to `ExecutionSpec` to disable those tensors without affecting outputs or training regularizers. There are no `recording` or `recording_fields` request arguments. Population signals and `projection.conductance` must be explicitly exposed to collect their diagnostic histories.

Custom specifications (`CustomNeuron`, `CustomSynapse`, `CustomInitializer`, `CustomConstraint` and custom operations/training helpers) refer to versioned definitions registered through `snnlab.extensions`. Bundles store names/configuration and required dependency names, not Python code.


### Explicit voltage sampling

Newly compiled graphs declare `voltage_sampling="explicit"`. Leaky integrators
expose `.pre_reset_voltage` after integration and before subtractive reset;
`.voltage` retains post-reset membrane diagnostics and continuation state.
`MeanVoltage` now explicitly reduces the pre-reset signal. Masks select time
samples, never phases, and sum/mean/final operations use their declared signal.
Graphs without the marker retain historical implicit unmasked pre-reset means
of integrator `.voltage`; loading never rewrites retained artifacts. Adopting
explicit sampling changes graph identity and requires compatible checkpoints.
Each call reduces its own measurement window; continuing dynamic state does
not implicitly carry cross-call reduction accumulators.
