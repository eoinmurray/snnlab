# Changelog

Changes to the combined snnlab Python distribution are recorded here. Package
versions use MAJOR.MINOR.PATCH. Component and serialized-schema versions identify
separate compatibility contracts.

## [Unreleased]

### Added

1. Added serialized `initialization_scaling="direct"` / `"fan_in_normalized"` controls to `Network.connect`, `Network.parameter` and `readouts.MeanVoltage`. Direct initialization preserves legacy readout weight scale without compensating initializer values or mutating tensors. Shared projection weights initialize once and reject conflicting policies. Older graphs retain normalized projection and direct operation defaults; checkpoint weights are loaded without rescaling. Updated API references and tutorial explanations.

2. Added opt-in `EpochObservations` and named `ObservationProbe` draws to graph epoch training: population rates in Hz, stored parameter L2 norms, output-population spike totals/silence/shares, and pre/post-clipping epoch gradient norms. Compact spike counting avoids additional dense observation histories; epoch evaluation preserves Python, NumPy, CPU and accelerator RNG streams. Audited checkpoints retain authenticated completed history and partial-epoch gradient totals, reject incompatible requests and do not fabricate missing past measurements. Updated request/result references and the Training tutorial.

3. Added explicit `CheckpointSelection` / `SelectionMetric` policies for completed-epoch graph-training checkpoints, including unweighted validation cross-entropy with accuracy tie-breaks. Selection evaluates the saved weights, records policy/scores/epoch/update and content-verified evaluation identity, and preserves the best candidate across checkpoint resume. Initial eligibility and exact ties are explicit; missing/non-finite metrics and incompatible resume requests fail. Added JSON CLI policy support, schema-version-4 epoch-selection checkpoints, request/result references and a Training tutorial example; the named legacy batch-loss default remains compatible.

4. Added typed `ExecutionSpec.interventions` (`DropSpikes`, `AddPoissonSpikes`, `ReplaySpikes`) and authenticated dense/sparse population-spike replay. Replay preserves ordinary neuron updates while replacing emitted spikes before downstream propagation, delay histories, recordings and readouts; absolute coverage supports exact continuation. Added pinned replay-file loading, CLI replay syntax, expected-intervention cache validation, bounded legacy/graph and continuation tests, and API/CLI documentation. The old `options["inference_interventions"]` and direct-forward intervention mappings now fail with migration guidance.

5. Dataset snapshot graph requests now automatically encode one batch at a time over verified temporary memory-mapped source arrays. Training gets fresh feature-scaled fixed/categorical Poisson draws; evaluation supports explicit repeatable validation draw sets. Versioned order/rate/spike streams and iteration coordinates reproduce resumed training and reject incompatible older protocols. Added automatic one-epoch/default-32-batch dataset behavior, batchwise inference with one model load, exact replay support, CLI categorical rates, memory/resume fixtures, and request/result/tutorial/CLI documentation.

6. Added typed absolute `BoundarySchedule`, selective `ResetVoltage` controls and online `DecisionSegments` to graph simulation/inference, with direct CLI JSON flags. Boundaries close the preceding segment before resetting the chosen leaky-integrator voltage and updating the next timestep; hidden, refractory, synaptic and delay state continue. Completed per-batch population totals and output-neuron spike counts appear in results and inference artifacts; compact partial counters and authenticated schedule metadata survive runtime save/load continuation. Reset/decision policy binds inference cache identity, with explicit mismatch checks and API/CLI documentation.

### Fixed

1. Newly authored graphs now use explicit leaky-integrator voltage sampling: `.voltage` remains post-reset membrane state and `.pre_reset_voltage` exposes the pre-reset integration sample. Masked and unmasked sum/mean/final operations consistently use the declared signal; `MeanVoltage` explicitly selects pre-reset voltage to preserve classifier behavior. Serialized `voltage_sampling="explicit"` distinguishes the corrected contract; older graphs without it retain historical reductions without rewriting checkpoints or artifacts. Added threshold-crossing, masks, gradients, continuation, diagnostics and legacy-compatibility tests, and documented phase and measurement-window semantics.

### Changed

1. Removed the root release-helper scripts and separate migration/versioning documents. Release preparation is manual, with instructions in the README; CI and publishing read the package version directly.

## [0.2.0] - 2026-10-04

### Changed

1. Removed Scira AI and Cursor from the documentation's Open menu.
2. `ExecutionSpec` now defaults to the graph executor. Typed requests requiring legacy routing must explicitly set `executor="legacy"`; the CLI retains its existing legacy default.
3. `ExecutionSpec` now exposes one `input_bindings` sequence accepting `DenseArrayBinding`, `EventStreamBinding`, `PoissonInputBinding` and `DatasetSnapshotBinding` through the public `InputBinding` type alias. Removed the separate `inputs`, `event_bindings`, `poisson_bindings` and `dataset_binding` constructor arguments; callers must migrate to typed bindings. Existing input compatibility rules and serialized execution protocols are retained.

4. `PoissonInputBinding.batch_size` now defaults to `1`. Required fields precede optional fields in its constructor; positional callers must migrate to the new order `(input_id, steps_count, rates_hz, seed, batch_size=1, categorical=False)` or use keyword arguments.

5. Replaced `ExecutionSpec.recording` and `recording_fields` with `diagnostics: bool = True`. Declared outputs always return; only explicitly exposed diagnostics return by default, and `diagnostics=False` disables them for simulation, inference and training. Renamed `ExecutionResult.recordings` to `diagnostics`, replaced the graph CLI `--recording` profile with `--diagnostics` / `--no-diagnostics`, and added `Projection.conductance` for explicit diagnostic exposure. Training regularizers and runtime continuation state remain independent of diagnostic retention.

6. The training example now learns both input-to-E and E-to-readout weights, with explicit fast-sigmoid surrogate gradients, separate learning rates and gradient clipping. Input weights remain constrained to be non-negative.

7. `ExecutionSpec` exposes `epochs`, `batch_size`, `shuffle`, `updates`, `save_final_checkpoint` and `save_selected_checkpoint` directly. Graph training rejects these settings inside `options`; callers must move them to constructor fields. The CLI adapter, examples and API reference use the direct fields. Inference-specific options remain in `options`.

8. The PyTorch Integration example now composes the bundled SNN with an external `Linear(2, 8) → ReLU → Linear(8, 2)` head. Both modules train together, share a saved state dictionary, and appear in the introductory diagram.

9. Consolidated input documentation: Network covers declarations, and ExecutionSpec covers binding types and compatibility rules. Removed the standalone Inputs reference from navigation; former routes redirect to ExecutionSpec.

10. `lang.LIF` now creates a supported current-based LIF specification (also available explicitly as `CUBA_LIF`), replacing the previously unsupported `lif` declaration. Projection weights and trace ports use synapse-specific units: `uS`/`.conductance` for conductance, `nA`/`.current` for current.

### Added

1. `ExecutionResult.numpy(batch=None)` returns named output and diagnostic arrays through `NumpyExecutionResult`, with an execution-derived `time_ms` axis. Optional batch selection uses declared signal axes, supporting both time-series and reduced outputs. Arrays are independent copies; original tensors and gradients remain intact. Quickstart now uses this API for plotting.

2. Added a runnable spike-pattern training example and Training documentation page with an explicit E-to-readout connection and named `w_out` weights, per-epoch training/validation loss and accuracy curves, and saved artifacts for later inference. `SpikeCount.parameters` now includes its readout weight so training recipes can select it directly.

3. Graph `train` now returns baseline and completed-epoch loss, accuracy and component metrics in `result.metrics["epochs"]`. `ExecutionSpec.validation` accepts a `ValidationSpec` containing held-out bindings and targets for evaluation without optimizer updates. Training now demonstrates one call handling all epochs and one final checkpoint save.

4. Added a paired Inference example and documentation page alongside Training. It loads Training's saved bundle and learned checkpoint, classifies fresh spike patterns without retraining, and saves predictions, checkpoint provenance, a network diagram and response plots.

5. Added a PyTorch Integration walkthrough and `examples/pytorch/training.py`. The example wraps Training's saved graph in an ordinary `nn.Module`, trains fresh weights using PyTorch data loaders and an external optimization loop, enforces graph constraints, plots per-epoch metrics, and saves/reloads a standard PyTorch state dictionary for test inference.

6. Expanded API Reference with Lang, Sim and Viz subgroups covering authoring, components, parameters, training recipes, compilation, operations/readouts, execution results, PyTorch integration, diagrams and plotting. Existing API URLs redirect to the grouped pages.

7. Added `CUBA_LIF`, `ExponentialCurrent` and the `nA` unit, with configurable rest/reset/initial voltage, refractory steps, surrogate gradients, training and runtime continuation. Current and conductance families are checked for compatibility.

8. Added versioned named registrations in `snnlab.extensions` for neurons, synapses, initializers, constraints, operations, objectives, regularizers, optimizers, surrogates and dataset encoders. Bundles retain definition/config dependencies without embedding code. Custom tensor state supports save/load continuation, regression objectives retain real targets, and external PyTorch loops can call `GraphExecutor.enforce_constraints()`.

9. Added a runnable Customisation example and documentation comparing standard current LIF and a registered adaptive neuron using matched weights, custom initialization, input/response/adaptation plots and an introductory diagram.

10. Added a standalone current-based LIF simulation tutorial and runnable example, with input, spikes, voltage and current plotted together; Customisation follows it with registered adaptive dynamics and a custom weight initializer.

### Fixed

1. Zero-delay feedforward connections between populations now retain a valid spike history for execution and runtime continuation.
2. `SignalLike.id` is now read-only, matching immutable `Signal` objects and readout properties. Network outputs and cross-entropy objectives share the same protocol, eliminating incorrect editor type errors for valid signals. Examples also check optional checkpoint and time-axis results before use.
3. Documentation navigation, search and LLM exports now use the current content collections instead of a module-level snapshot. Content and sidebar metadata changes invalidate development routes so newly added pages appear without restarting the server.

## [0.1.1] - 2026-10-04

### Added

1. Static Fumadocs documentation with KaTeX, search, GitHub Pages and Cloudflare hosting.
2. Complete source-derived API references for `lang`, `sim` and `viz`, with automated drift checks.
3. Six runnable general examples for bundles, simulation, input replay, training, checkpoint resume and retained-signal plotting.
4. A changelog, single-source package version and release preparation/check helper.
5. Automatic PyPI publishing on version-source changes to `main`, using Trusted Publishing and release tags after package checks pass.

### Changed

1. The documentation opens at the site root; previous `/docs/` paths redirect there.
2. Documentation builds use Astro static output with Fumadocs React islands in place of Next.js.
3. Package builds read the runtime version directly, and source archives exclude local documentation dependencies, caches and generated output.

## [0.1.0] - 2026-10-03

### Added

1. Initial combined Python distribution extracted from Pinglab, exposing `snnlab.lang`, `snnlab.sim` and `snnlab.viz`.
2. Portable graph authoring and validated bundles, graph-native and legacy execution, surrogate-gradient training, retained artifacts and visualization utilities.
3. Portable tests and package checks, retaining existing numerical defaults, schemas and backend identifiers.

The historical baseline is Git tag `v0.1.0`. Changes above it remain unreleased
until a subsequent version is prepared and tagged.
