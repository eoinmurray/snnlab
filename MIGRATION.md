Initial extraction from Pinglab working tree at eb77f161104155eae31b44986ddb3f45f1d33e24.

Includes the existing uncommitted simulator execution changes and the two new COBA-threshold/spiking-readout tests present at extraction. Changes in this extraction are packaging, imports, module entry points and the default training output path (now relative to the invoking working directory). Scientific schemas and numerical defaults are retained.

The pre-existing threshold test used an unsupported initial_voltage_mv field. Its fixture now uses an identical conductance pulse to place the two populations on opposite sides of their thresholds; simulator dynamics remain unchanged.

## Unified execution input bindings

`ExecutionSpec` now accepts all input sources through `input_bindings`. Replace the removed constructor arguments as follows:

1. `inputs={"name": tensor}` becomes `input_bindings=(DenseArrayBinding("name", tensor),)`.
2. `event_bindings=(events,)` becomes `input_bindings=(events,)`.
3. `poisson_bindings=(poisson,)` becomes `input_bindings=(poisson,)`.
4. `dataset_binding=snapshot` becomes `input_bindings=(snapshot,)`.

Combine dense/event bindings for different inputs in the same sequence. Poisson bindings remain exclusive with replay; a dataset snapshot must be the sole binding. Serialized graph, checkpoint and execution-protocol formats are unchanged.

## Graph diagnostics

Declare essential results with `net.output` and optional diagnostic signals with `net.expose`. `ExecutionSpec` now returns exposed diagnostics by default; use `diagnostics=False` to disable them. Retrieve them through `result.diagnostics`, replacing `result.recordings`.

Remove `recording` and `recording_fields` arguments. Previously automatic full traces must now be explicitly exposed: use population `.spikes`/`.voltage` signals or a returned projection's `.conductance`. The graph CLI uses `--no-diagnostics` to disable diagnostics; `--recording` is removed. Outputs, training regularizers and runtime state remain available independently.

## Direct training controls

Pass `epochs`, `batch_size`, `shuffle`, `updates`, `save_final_checkpoint` and
`save_selected_checkpoint` directly to `ExecutionSpec`. Graph training rejects
these keys inside `options` instead of silently ignoring them. Defaults preserve
the existing full-batch mode (`epochs=0`), while a positive epoch count enables
minibatch training. Inference settings remain in `options`.

```python
execution = ExecutionSpec(
    kind="train", bundle=bundle_path, input_bindings=(train_input,),
    targets={"class": labels}, epochs=20, batch_size=32, shuffle=True,
)
result = train(execution)
```

## Current-based dynamics and registered extensions

`lang.LIF(...)` now produces current-based LIF dynamics; `lang.CUBA_LIF(...)` is the explicit equivalent. Connect it with `lang.ExponentialCurrent(tau=...)` using `nA` parameters. Existing COBA graphs retain `uS` parameters and conductance synapses. Use `.current` or `.conductance` according to the projection family.

Custom definitions require explicit imports of their registration modules before compilation or execution. Bundles store versioned names and configuration only. Runtime-state artifacts containing current or custom state use manifest version 2; old version 1 artifacts remain readable. Custom optimizer checkpoints support per-parameter tensors/scalars, not arbitrary Python objects or scheduler/global state.
