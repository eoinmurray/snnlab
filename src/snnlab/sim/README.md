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
from snnlab.sim import ExecutionSpec, build

built = build(ExecutionSpec(
    kind="build",
    bundle=Path("network.bundle"),
    device="cpu",
    seed=17,
))
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

1. `ExecutionSpec` configures a build, simulation, inference, or training request.
2. `build(spec)` constructs the model and returns it in `ExecutionResult.model`.
3. `simulate(spec)` resolves input bindings and runs the graph.
4. `infer(spec)` uses the simulation handler, including checkpoint loading.
5. `train(spec)` runs training with the supplied recipe and targets.
6. `execute_request(spec)` dispatches according to `spec.kind`.
7. `GraphRuntimeState` carries simulation state between compatible runs.
8. `ExecutionResult` holds outputs, diagnostics, selected recorded signals, decisions,
   metrics, parameters and runtime state. `.numpy()` converts outputs/diagnostics;
   convert selected `recorded_signals` tensors separately.

See the full [ExecutionSpec reference](../../../docs/content/docs/api/sim/execution-spec.mdx)
and [ExecutionResult reference](../../../docs/content/docs/api/sim/execution-result.mdx)
for fields, input bindings, and result handling.

## Source organisation

`__init__.py` is the only public export location. The packages below are internal
implementation details; applications should import supported functions and types
from `snnlab.sim`. Each package initializer is deliberately minimal.

| Package | Responsibility and source files | API reference |
| --- | --- | --- |
| `requests/` | [spec.py](requests/spec.py), [result.py](requests/result.py), [handlers.py](requests/handlers.py): requests, results, build/simulate/infer and dispatch. | ExecutionSpec, ExecutionResult, Execution Functions |
| `inputs/` | [bindings.py](inputs/bindings.py), [resolution.py](inputs/resolution.py), datasets, encoders and providers. | ExecutionSpec input bindings and resolution |
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

`ExecutionSpec` and `GraphExecutor` accept `compile_mode=None` (eager),
`"default"`. Built-in neuron/readout kernels compile;
Python scheduling and custom callbacks remain eager. Compiler warmup and
floating-point fusion must be assessed for the intended workload.

`ExecutionSpec(dataset_prefetch=2)` overlaps built-in CPU dataset encoding and
CUDA transfers through a bounded queue; the default is 0. CPU seeds and batch
order remain unchanged. Custom encoders retain caller-thread execution.
`RecordingSpec(block_steps=32)` groups selected samples before CPU transfers;
the default is 1 for existing sinks. Partial blocks flush at call completion.

Inference defaults to `timing_mode="synchronized"`, reporting completed
input-resolution/execution time as `simulate_s`. `"enqueue"` reports host time
as `simulate_enqueue_s` and omits explicit fences. `peak_python_bytes` is Python
memory only. See the [performance reference](../../../docs/content/docs/api/sim/execution-spec.mdx)
and `benchmarks/gpu_efficiency.py` for verification and measured limitations.

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
`ExecutionSpec(kind="train", ...)`. Import `TraceSpec` from `snnlab.sim`.
Modes `off`, `monitor`, `profile` and `both` support bounded selected-update
measurements, detached activity/sparsity, phase timing, memory and PyTorch
CPU/CUDA timelines. Tracing is disabled by default and does not change checkpoint
identity. Hardware counters unavailable from PyTorch remain explicitly unavailable.
See [Tracing](../../../docs/content/docs/api/sim/tracing.mdx) and the
[SHD benchmark](../../../benchmarks/shd/README.md) for scope and CLI controls.

## Run snapshots

Set `ExecutionSpec(snapshot="runs/example", ...)` to save `snapshot.json` and
optional `parameters.npz` before simulation, inference or training. Use
`SnapshotSpec(path="runs/example", stage="both", parameters=True)` for separate
initial/final exports. Captured `RunSnapshot` objects are in `result.snapshots`;
`write_run_snapshot(path, snapshot)` re-exports them. Snapshots preserve resolved
settings and copied weights, independently of `retain_parameters`; they do not
replace training checkpoints or runtime state. Existing exports are not overwritten.
