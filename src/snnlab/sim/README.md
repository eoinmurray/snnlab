# snnlab.sim

`snnlab.sim` executes spiking-network graphs and supports surrogate-gradient
training. Import the supported Python API from `snnlab.sim`. Its central export
map in `__init__.py` resolves implementations lazily, keeping public imports
independent of the internal file layout. The former `snnlab.sim.execution`
module has been retired; import its supported functions and types from
`snnlab.sim`.

Explicit source arrays and replay spikes are available through `load_dataset`,
`encode_images_poisson` and `poisson_spikes`. The spike generators return binary
Bernoulli occupancy rather than Poisson counts; timesteps use ms and rates Hz.
`fast_sigmoid_spike`, `lif_step_expeuler` and `init_lif_state` are supported
numerical reference helpers. Import all six names from `snnlab.sim`, rather than
their implementation modules. See the [source and reference contracts](../../../docs/content/docs/api/sim/execution-functions.mdx).

Physical-time helpers are available from `snnlab.lang`. Fresh historical COBA LIF
replay graphs use `lang.compile(net, refractory_timing="legacy_coba")`; the
compiler authenticates the selected schedule and any training recipe together.
See [Compilation](../../../docs/content/docs/api/lang/compilation.mdx) for exact
duration, counter/held-step and checkpoint compatibility rules.

Built-in spiking models include current and conductance LIF, ALIF and AdEx.
ALIF uses a spike-triggered threshold offset; AdEx uses an exponential voltage
term and adaptation current. Their per-cell adaptation is differentiable and
included in saved runtime state and recordings. See the
[Neurons guide](../../../docs/content/docs/neurons.mdx) for parameters and
the exponential-Euler discretization.

## GraphExecutor

`GraphExecutor` is a PyTorch `torch.nn.Module` that executes a prepared
`GraphPlan`. There is no type named `GraphExecution`.

```python
from snnlab.sim import GraphExecutor, plan_graph

# graph is a validated graph mapping, such as a compiled bundle's graph.json.
model = GraphExecutor(plan_graph(graph), seed=17)

# inputs maps graph input IDs to tensors with (time, batch, *features) axes.
result = model(inputs)
```

`plan_graph(graph)` lowers graph topology before simulation. The executor
initializes parameters and advances the network through the supplied inputs.
Its constructor accepts `seed`, `trainable_parameters`, `surrogate_slope`, and
an optional `surrogate` mapping. Use `.to(device)` to move the model to a
PyTorch device, and supply input tensors on that device.

Calling the model returns an `ExecutionResult`. The call accepts the keyword
arguments `diagnostics`, `runtime_state`, `interventions`, `resets`, `decisions`,
`measurement`, `recording` and `retain_outputs`. To continue a
simulation, pass a compatible previous result's `runtime_state` into the next
call. `parameter_map()` exposes parameters by graph parameter ID;
`enforce_constraints()` applies declared constraints after an external
optimizer step.

## Execution requests

For bundle loading, input resolution, device selection, and execution metadata,
use the request API:

```python
from pathlib import Path
from snnlab.sim import build

built = build(
    bundle=Path("network.bundle"),
    device="cpu",
    seed=17,
)
model = built.model
```

The bundle path must point to an existing compiled graph bundle. Building
initializes the model without running inputs.

Missing bundle paths raise `FileNotFoundError` with `bundle not found: ...`;
the CLI reports the same error. Saved custom definitions are automatically
loaded from their external `module:function` references. Make those modules
available in each execution environment. Graphs and recipes contain callback
locations and optional exact package versions, with no Python source or pickled
functions. Missing implementations and incompatible declared versions fail
before execution. See the [Extensions reference](../../../docs/content/docs/api/lang/extensions.mdx).

1. `build(...)` loads and constructs a graph executor.
2. `simulate(...)` resolves input bindings and runs the graph.
3. `infer(...)` uses the simulation handler, including checkpoint loading.
4. `train(...)` runs training with the supplied recipe and targets.
5. Public functions take explicit keyword arguments; the CLI and handlers share an internal `ExecutionSpec`.
6. `GraphRuntimeState` carries simulation state between compatible runs.
7. `ExecutionResult` holds outputs, diagnostics, selected recorded signals, decisions,
   metrics, parameters and runtime state. `.numpy()` converts outputs/diagnostics;
   convert selected `recorded_signals` tensors separately.

See the full [Execution Options reference](../../../docs/content/docs/api/sim/execution-spec.mdx)
and [ExecutionResult reference](../../../docs/content/docs/api/sim/execution-result.mdx)
for fields, input bindings, and result handling.

## Source organisation

`__init__.py` is the only public export location. The packages below are internal
implementation details; applications should import supported functions and types
from `snnlab.sim`. Each package initializer is deliberately minimal.

| Package | Responsibility and source files | API reference |
| --- | --- | --- |
| `requests/` | [api.py](requests/api.py), [spec.py](requests/spec.py), [result.py](requests/result.py), [handlers.py](requests/handlers.py): requests, results, build/simulate/infer and dispatch. | Internal requests, ExecutionResult, Execution Functions |
| `inputs/` | [bindings.py](inputs/bindings.py), [resolution.py](inputs/resolution.py), datasets, encoders and providers. | Input bindings and resolution |
| `graph/` | [plan.py](graph/plan.py), [executor.py](graph/executor.py), dynamics, tensor kernels and extension helpers. | GraphExecutor, PyTorch Integration |
| `runtime/` | [state.py](runtime/state.py), [recording.py](runtime/recording.py), [interventions.py](runtime/interventions.py), [resets_decisions.py](runtime/resets_decisions.py). | Runtime State, Recording and Measurement, Interventions, Resets and Decisions |
| `training/` | [trainer.py](training/trainer.py), policies, observations, selection and [checkpoints.py](training/checkpoints.py). | Execution Functions, training policies, Training Checkpoints |
| `artifacts/` | [bundle.py](artifacts/bundle.py), [inference.py](artifacts/inference.py), [integrity.py](artifacts/integrity.py): loading, inference products and shared canonical digests. | Compilation, Inference Artifacts |
| `diagnostics/` | [tracing.py](diagnostics/tracing.py), [conformance.py](diagnostics/conformance.py): bounded training diagnostics and internal numerical comparison tools. | Tracing |

Shared, tensor-free contracts live outside the simulator in `snnlab/_shared`:
[neurons.py](../_shared/neurons.py) validates built-in neuron specifications,
[connectivity.py](../_shared/connectivity.py) resolves shared structural rules,
and [timing.py](../_shared/timing.py) converts physical durations to timesteps.
Both authoring and execution consume these standard-library-only modules;
they do not depend on `lang` or `sim`. The public extension registry remains
[snnlab/extensions.py](../extensions.py).

Request handlers orchestrate execution. Graph dynamics and declaration/result
modules do not import those handlers or the native trainer. Format identifiers
live beside their formats, graph metadata helpers live with planning, and
training reporting helpers live with the trainer. Command-line argument
adaptation is kept in `cli.py`.

The documentation site's [GraphExecutor API reference](../../../docs/content/docs/api/sim/graph-executor.mdx)
appears under API Reference → Sim.

## Command line

```sh
python -m snnlab.sim --help
python -m snnlab.sim sim --help
```

The installed `snnsim` command provides the same CLI. See the
[CLI reference](../../../docs/content/docs/api/sim/cli.mdx).

## GPU performance controls

Execution functions and `GraphExecutor` default to `compile_mode="default"`.
Built-in/custom neuron and synapse updates, custom graph operations, surrogate
derivatives and post-update weight constraints compile. Python scheduling,
initialization and training orchestration stay eager. Use `compile_mode=None`
or CLI `--compile-mode eager` to disable compilation. Callback signatures stay
unchanged; write vectorized torch operations with stable tensor state and avoid
host scalar reads, NumPy conversions and tensor-dependent Python branches.
Strict custom surrogate diagnostics retain an eager autograd boundary.
Measure compiler warmup and compare outputs/gradients against eager execution;
fused floating-point arithmetic can differ slightly.

`dataset_prefetch=2` overlaps built-in CPU dataset encoding and
CUDA transfers through a bounded queue; the default is 0. CPU seeds and batch
order remain unchanged. Custom encoders retain caller-thread execution.
`RecordingSpec(block_steps=32)` groups selected samples before CPU transfers;
the default is 1 for existing sinks. Partial blocks flush at call completion.

Inference defaults to `timing_mode="synchronized"`, reporting completed
input-resolution/execution time as `simulate_s`. `"enqueue"` reports host time
as `simulate_enqueue_s` and omits explicit fences. `peak_python_bytes` is Python
memory only. See the [performance reference](../../../docs/content/docs/api/sim/execution-spec.mdx)
and `benchmarks/execution/benchmark.py` for verification and measured limitations.

For opt-in network compilation use `compile_block_steps=16` (1-64; default 1).
Projection, synapse, ordered built-in neuron/readout, spatial and delay work then
shares a bounded tensor graph. Internal blocks retain gradients and final partial
blocks are included. `execution_path` reports the selected path and fallback
reasons; native training retains all observed paths in `execution_paths`.
Recording, online reductions, registered dynamics/surrogates, interventions,
reset/decision schedules, tracing and incompatible parameter dependencies use the
individual-kernel loop. `compile_mode=None` remains fully eager.

Runtime floating state follows model dtype, including recurrent `.half()` and
`.bfloat16()` execution with matching inputs. Counters remain integer; MPS cannot
use float64. Precision changes remain explicit. Built-in Poisson inputs generate
spikes on the selected device from compact rates/features. Accelerator encoding
records `snnlab.poisson-device/v1`; CPU draws retain their stream, while
cross-device bitwise equivalence is not promised. Event datasets are indexed once
and vectorized per sample batch. Custom encoders can accept `device="cpu"` to
opt into device preparation; legacy callbacks keep their arguments. Prefetched
callbacks run on the caller thread under scoped RNG state.

Replay transfers precede stepping. CUDA recordings use bounded asynchronous
pinned-host chunks; decision reporting transfers counts after stepping. Run
`uv run python benchmarks/execution/benchmark.py --devices cpu` for matched
numerical gates, cold/warm complete-work timings and explicit measurement limits.
Retained CPU/CUDA/MPS measurements and fixed-shape CUDA Graph training replay
experiments are documented in [the benchmark guide](../../../benchmarks/execution/README.md).
CUDA Graph replay remains a standalone experiment. Cell-selection indices for
recordings are staged before stepping; recording all cells avoids index transfers.

## Projection arithmetic and snapshots

Graph execution defaults to `projection_mode="fast"`: built-in projections with
exactly zero numerical decay use current drive directly. `"strict"` opts into
historical IEEE multiplication by zero. Fast mode can recover finite drive after
nonfinite projection state and removes its zero backward dependency. Custom
synapse equations and nonzero recurrences remain unchanged.

Direct calls accept independent `retain_parameters`, `retain_final_state` and
`retain_runtime_state` flags, all defaulting to `True`. Fresh-batch training omits
unused forward snapshots automatically. Request runtime state for continuation.
Numerical mode and retention bind checkpoint/cache compatibility; historical
zero-decay states and checkpoints use strict arithmetic explicitly. See the
execution reference for migration and nonfinite/gradient semantics.

## Optional training tracing

Pass `tracing=TraceSpec(output=Path("new-trace-directory"), mode="both")` to
`train(tracing=..., ...)`. Import `TraceSpec` from `snnlab.sim`.
Modes `off`, `monitor`, `profile` and `both` support bounded selected-update
measurements, detached activity/sparsity, phase timing, memory and PyTorch
CPU/CUDA timelines. Tracing is disabled by default and does not change checkpoint
identity. Hardware counters unavailable from PyTorch remain explicitly unavailable.
See [Tracing](../../../docs/content/docs/api/sim/tracing.mdx) and the
[SHD benchmark](../../../benchmarks/shd/README.md) for scope and CLI controls.

## Run snapshots

Set `snapshot="runs/example"` in an execution call to save `snapshot.json` and
optional `parameters.npz` before simulation, inference or training. Use
`SnapshotSpec(path="runs/example", stage="both", parameters=True)` for separate
initial/final exports. Captured `RunSnapshot` objects are in `result.snapshots`;
`write_run_snapshot(path, snapshot)` re-exports them. Snapshots preserve resolved
settings and copied weights, independently of `retain_parameters`; they do not
replace training checkpoints or runtime state. Existing exports are not overwritten.
