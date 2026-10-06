# snnlab.sim

`snnlab.sim` executes spiking-network graphs and supports surrogate-gradient
training. The Python graph-execution API lives in `snnlab.sim.execution`;
these types and functions are not re-exported from `snnlab.sim`.

## GraphExecutor

`GraphExecutor` is a PyTorch `torch.nn.Module` that executes a prepared
`GraphPlan`. There is no type named `GraphExecution`.

```python
from snnlab.sim.execution import GraphExecutor, plan_graph

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
from snnlab.sim.execution import ExecutionSpec, build

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
for fields, input bindings, and result handling. Implementation details are in
[execution.py](execution.py).

The documentation site's [GraphExecutor API reference](../../../docs/content/docs/api/sim/graph-executor.mdx)
appears under API Reference → Sim.

## Command line

```sh
python -m snnlab.sim --help
python -m snnlab.sim sim --help
```

The installed `snnsim` command provides the same CLI. See the
[CLI reference](../../../docs/content/docs/api/sim/cli.mdx).
