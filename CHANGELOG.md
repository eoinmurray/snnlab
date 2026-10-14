# Changelog

Changes to the combined snnlab Python distribution are recorded here. Package
versions use MAJOR.MINOR.PATCH. Component and serialized-schema versions identify
separate compatibility contracts.

## [Unreleased]

## [0.13.0] - 2026-10-14

### Added

1. `training.FastSigmoid` now accepts independent `gradient_scale` and `threshold_inclusive` settings. Training bundles and direct executors preserve the complete configuration, allowing the SHD derivative with slope `100/15`, scale `1/100` and strict threshold comparison to use compiled built-in neuron kernels. Non-default scales use an exact hard forward with an explicit tensor-only backward, avoiding CUDA fusion residuals from a scaled detach proxy. Exclusive built-in thresholds bind runtime-state compatibility; resolved surrogate and compilation settings appear in execution metrics. Defaults retain historical forward/backward arithmetic and serialized recipe identity; registered derivatives still execute eagerly.
2. Added a synthetic SHD-shaped COBA/delayed-PING profiling harness and Modal H100 launcher under `benchmarks/shd-compilation/`. It separates cold updates, warmed forward/backward/update timings, allocated GPU memory and CPU/CUDA profiles across batches 32, 256 and 8. On one H100, batch-256 synthetic 1,000-step updates measured 2.718 → 1.716 s for registered-to-compiled COBA and 5.582 → 3.242 s for registered-to-compiled delayed PING, with higher cold costs. Raw samples, profiles and limits are retained alongside the harness. This prepares timestep fusion; no timestep-block path or general speedup is established.

3. Added opt-in `ExecutionSpec.training_timing` and CLI `--training-timing` for graph training. Results report disjoint setup, training, training-split evaluation, validation/probe, final checkpoint and reporting durations, plus per-epoch phase durations. Synchronized mode fences CUDA/MPS at phase boundaries; enqueue mode reports host latency. Disabled timing adds no timing clocks or fences. Timings cover only work performed in the current call and do not alter checkpoint, recipe or resume identity.

4. Added a dedicated MNIST snnTorch/SNNLab comparison page with matched architecture and training settings, run-pinned learning curves, the complete timing table, measurement definitions and downloadable figure/raw results. The report retains the independently trained accuracy gap and limits its speed claims to the measured workload.

### Changed

1. The MNIST snnTorch comparison now measures matched setup/training/initial-and-epoch-evaluation/in-memory-checkpoint scopes, with disk writes and Modal volume commits outside both timers. SNNLab uses one public `train()` call; isolated forward/backward profiling remains separate from training throughput. The benchmark retains three code files and three result files.

2. Preserved the retired zero-decay model helpers and reference data under test fixtures, retaining sustained training acceptance coverage after benchmark cleanup. The current MNIST harness accepts newer releases while requiring its projection/snapshot and training-timing APIs.

### Validation

1. The final focused local suite passed 115 tests (15 CUDA-dependent skips). The H100 regression suite passed 51 checks, including actual CUDA compilation, exact binary spikes, trajectories, gradients, Adam state and runtime continuation. Small spiking COBA/PING probes passed at `atol=rtol=2e-5`; full 1,000-step training-trajectory equivalence is not established. Lint, documentation type checks and both root/GitHub Pages builds passed. The remaining timestep-fusion work is tracked in [22](https://github.com/eoinmurray/snnlab/issues/22); the configurable-surrogate and profiling portion of [21](https://github.com/eoinmurray/snnlab/issues/21) is complete.

2. Added timing opt-in/default, synchronized/enqueue, partial-minibatch and single-batch checks, including exact unchanged losses, gradients, final parameters, Adam state and checkpoint/RNG protocol checks.
3. Concurrent 20-epoch MNIST training on Modal L4 completed 940 updates per library. Matched total workflow times were 162.39 s for snnTorch and 148.61 s for SNNLab; training-loop times were 59.94 s and 53.46 s. Both libraries' losses, accuracies and update histories exactly matched their preceding untimed run. Final test accuracy remained 85.20% versus 80.01%, so timing validation does not establish cross-library training equivalence. Separate randomized profiling retained 900 raw samples and verified identical output spikes at all three shared weight states.

## [0.12.1] - 2026-10-14

### Validation

1. Added a float32 spiking-classifier regression comparing scores, losses, gradients, weights and Adam moments across 40 clipped updates in strict and fast modes, on CPU and CUDA with matched backward operations. Full zero-decay benchmark runs now independently gate epoch losses, test loss and test accuracy; the retained cross-library accuracy discrepancy must fail that gate despite passing shared-checkpoint comparisons. Diagnostic artifacts are saved before rejection.

2. Retained the training accuracy investigation: matching surrogate backward division and bias-gradient matrix reductions produced identical reported two-epoch losses and 87.20% accuracy for both libraries. This diagnostic intervention changes no released library arithmetic. The focused regressions passed six local CPU cases and both CUDA cases on Modal L4.
3. Repeated the original, unmatched two-epoch MNIST comparison across five preselected seeds and all four conditions (20 completed Modal L4 trials). Mean accuracy was 76.64% for snnTorch and 80.99% for SNNLab; SNNLab scored higher in four of five seeds. All SNNLab arithmetic/snapshot variants had identical losses, accuracies and final weights per seed. The original 6.25-point SNNLab deficit did not persist consistently; the wide exploratory paired interval includes zero, so five seeds establish neither superiority nor equivalence. Raw results, analysis, figures and reproducible launchers are retained under `benchmarks/zero-decay/`. Concurrent trial durations are not performance comparisons.

## [0.12.0] - 2026-10-14

### Changed

1. Graph execution now defaults to `projection_mode="fast"` for training, inference and direct executor calls. Built-in projections with exactly zero numerical decay use current drive directly; custom synapse callbacks and nonzero recurrences retain their equations. `"strict"` opts into the historical `previous * decay + drive` expression. Fast mode can clear stale nonfinite projection state and removes its zero backward dependency; signed-zero behavior and absent-versus-zero leaf gradients can differ. Custom extension validation remains a separate policy.
2. Added independent public `retain_parameters`, `retain_final_state` and `retain_runtime_state` controls, with existing full snapshot defaults preserved. Fresh-batch training/evaluation omit all unused forward snapshots automatically. Returned training parameters respect retention, while mandatory optimizer checkpoints remain complete. CLI arithmetic/retention switches match the Python API and reject saving an omitted runtime snapshot.
3. Fast arithmetic is recorded in training protocols and inference retention identity. Resuming across arithmetic modes fails explicitly; historical checkpoint protocols denote strict arithmetic. Runtime compatibility binds fast semantics for enabled built-in zero-decay projections, so their older continuation states require strict execution. Retention identities bind omitted snapshots. Persisted schemas and graph identities are unchanged. Resolves [20](https://github.com/eoinmurray/snnlab/issues/20).

### Validation

1. Added CPU/CUDA finite-output, trajectory, gradient, optimizer-update and continuation comparisons, nonfinite forward/backward cases, custom-synapse routing, independent snapshot controls, clone-work checks, dataset/cache retention and historical-strict checkpoint resume coverage. Added a controlled matched Synaptic/snnTorch benchmark with initial/newly-trained checkpoints, randomized synchronized timings, separate profiles and complete MNIST training/test processing. Measured results and limitations are retained under `benchmarks/`.
2. On one Modal L4, trained batch-128 warmed steps fell from 129.36 ms strict to 111.69 ms fast. Two full MNIST epochs fell from 103.94 s to 89.40 s, or 88.01 s with output-only results. All three SNNLab runs produced identical reported losses and 77.67% test accuracy. Fixed-weight comparisons with snnTorch passed, while its independently trained run reached 83.92%; full cross-library training equivalence is not established. The isolated release suite passed 838 CPU tests (14 accelerator skips), with 76 remote checks and 20 separate CPU portability checks.

## [0.11.0] - 2026-10-14

### Changed

1. Training snapshots batch-loss selection candidates only when they improve. Internal training/evaluation forwards omit unused parameter snapshots, gradients are copied only for the final result or requested observations, and component/evaluation reporting batches host reads. Final and selected checkpoints, optimizer/RNG resume, data order and pre-clipping gradient reporting retain their contracts. Batch-loss selection and requested gradient-norm observations still require host decisions/measurements. Resolves [14](https://github.com/eoinmurray/snnlab/issues/14) and [15](https://github.com/eoinmurray/snnlab/issues/15).
2. Added opt-in `compile_mode` to graph executors, execution requests and the CLI. Prepared built-in neuron/readout kernels support `torch.compile`; projection products, graph scheduling and custom callbacks remain eager. Custom surrogates retain eager neuron dispatch. Compilation warmup, new-shape variants and fused floating-point differences must be assessed for each workload. Resolves [17](https://github.com/eoinmurray/snnlab/issues/17).
3. Added bounded `dataset_prefetch` for built-in CPU dataset encoders in training, evaluation and inference, with CUDA pinned transfers, event/stream ownership and caller-thread fallback for custom encoders. Existing CPU random streams, batch order, metadata and resumed updates are preserved. Default 0 retains synchronous preparation. Resolves [16](https://github.com/eoinmurray/snnlab/issues/16).
4. Added `RecordingSpec.block_steps` and recording JSON support for bounded timestep blocks. Event extraction runs on transferred CPU blocks; partial blocks flush at call completion. Default 1 preserves sink granularity and historical retention identities. Larger blocks opt into grouped callbacks and higher bounded device memory use. Resolves [18](https://github.com/eoinmurray/snnlab/issues/18).
5. Inference `timing_mode="synchronized"` now reports completed input-resolution/execution time as `simulate_s`, excluding build/checkpoint loading. Opt-in `"enqueue"` omits timing fences and reports `simulate_enqueue_s`. Timing mode/scope are explicit; `peak_python_bytes` remains Python-only allocation memory. Resolves [19](https://github.com/eoinmurray/snnlab/issues/19).

### Validation

1. Added CPU and CUDA checks for all six built-in spiking models, compiled outputs/gradients/optimizer updates, recurrent delays and continuation, deterministic prefetch/resume, selected dense/event recording blocks, discarded checkpoint candidates and timing boundaries. Added reproducible CUDA and Modal benchmark scripts, a baseline comparison against 0.10.1, and recorded target-GPU results under `benchmarks/`. Warm throughput, cold compiler cost and allocated memory are reported separately; reduced copy/synchronisation counts do not imply an eager full-training speedup.

## [0.10.1] - 2026-10-14

### Changed

1. Structural custom-extension validation is now the default for `ExecutionSpec`, `GraphExecutor` and the CLI. Tensor type, shape, device, dtype and state-key checks remain enforced; initialization and restored runtime state remain strict. Per-step finite-value/binary-spike checks and custom-surrogate derivative finite checks require explicit `extension_validation="strict"` or `--extension-validation strict`. This changes the default introduced in 0.10.0: invalid callback values can propagate silently unless strict diagnostics are enabled. Saved formats and numerical equations are unchanged.

## [0.10.0] - 2026-10-14

### Changed

1. Custom extension state validation scans each tensor for finiteness once, rather than twice. Graph executors resolve custom surrogate dispatch once and reuse its autograd function without changing derivative equations.
2. Added `extension_validation="structural"` to `ExecutionSpec` and `GraphExecutor`, and `--extension-validation structural` to the CLI. This opt-in policy retains tensor type/shape/device/dtype/state-key contracts while skipping custom neuron/synapse step finite checks, binary-spike checks and custom surrogate backward finite checks. Strict diagnostics remain the default; initialization and runtime-state restoration remain strict under both policies. Build/run metrics record the policy. No zero-decay numerical semantics or persisted formats changed.

## [0.9.0] - 2026-10-13

### Added

1. Added projection-level `allow_autapses=False` to `Network.connect` and returned `Projection` objects. Newly authored same-population connections exclude self-feedback by default; `True` deliberately enables it. Identity determines exclusion, so equally sized different populations keep matching-index connections. Graphs and bundles serialize the requested policy; initialization/run metadata, native PyTorch checkpoints and runtime continuation record its effective policy and eligible fan-in. Shared weights with incompatible structural masks fail validation.

### Changed

1. Autapse-free initialization calculates sparsity and fan-in normalization from eligible other neurons. Exact-K samples distinct eligible partners directly and retains ties-to-even rounding and the minimum-one rule when partners exist; a single-neuron population receives no recurrent connections. Direct scaling still skips only fan-in division. Execution and gradient masks, post-update constraints and checkpoint restoration preserve exclusion without freezing other initially zero weights.

### Compatibility

1. Historical graphs/bundles without `allow_autapses` retain permissive initialization, random draws, normalization, graph identity and checkpoint metadata. Native checkpoints without policy metadata remain loadable with their historical graphs; loading them into a graph that excludes autapses fails explicitly. Changing a declared policy invalidates checkpoint and runtime continuation compatibility. Updated connectivity/weights documentation and added E→E/I→I initialization, optimization, persistence, shared-weight and boundary tests. Downstream experiment reporting remains tracked in [Pinglab issue 90](https://github.com/eoinmurray/pinglab/issues/90).

## [0.8.0] - 2026-10-13

### Changed

1. Breaking authoring change: all built-in LIF, ALIF and AdEx neurons define refractory periods with `refractory=duration * lang.ms`. `refractory_steps` and `refractory_policy` are unsupported constructor arguments. Duration conversion supports exact timestep multiples only; the `nearest` policy and policy arguments to timing helpers have been removed. Zero disables the refractory hold, while non-divisible, negative and non-finite durations fail before execution. Physical defaults remain zero for CUBA LIF and ALIF/AdEx, and 3 ms or 1.5 ms for COBA LIF according to its membrane time constant.
2. Newly compiled graphs serialize `refractory_timing="following_steps"` and all built-in spiking neurons suppress the derived number of following steps while holding reset voltage. This adds one held timestep to new COBA LIF graphs compared with the historical counter convention. Runtime counters are resolved once during planning; result/build metrics record requested and realised ms, derived steps, actual held duration and scheduling. Timestep inference overrides preserve source physical durations or fail exact-grid validation.

### Compatibility

1. Authenticated historical bundles retain their original identity, counts, COBA scheduling and compatible saved runtime state without rewriting files. Added `lang.migrate_refractory(graph)` to create a validated duration-only copy using the original timestep and retaining legacy COBA scheduling. Conflicting duration/count representations fail. Migrated copies have new graph identities; checkpoints and runtime state are not silently converted or rebound. Removed legacy configs, CLI switches and predefined E/I components remain removed.
2. Updated neuron/API documentation and regenerated the neuron comparison, Quickstart and Training → Inference/PyTorch example figures under the new scheduling. E/I test circuits now use explicit 3 ms/1.5 ms durations; authenticated historical forward/backward fixtures continue to exercise their original graph contracts. Added coverage across all six models for exact conversion, common scheduling, zero/invalid durations, bundle migration, saved continuation and timestep overrides.

## [0.7.0] - 2026-10-07

### Added

1. Built-in ALIF and AdEx neurons for current and conductance synapses: `ALIF`/`CUBA_ALIF`, `COBA_ALIF`, `ADEX`/`CUBA_ADEX` and `COBA_ADEX`. ALIF adapts its firing threshold; AdEx combines exponential spike initiation with subthreshold and spike-triggered adaptation current. Parameters use explicit ms/mV/nF/uS/nA units. Per-cell adaptation supports outputs, diagnostics, streamed recordings, surrogate-gradient training and saved runtime continuation. Existing LIF defaults and bundle format versions are retained.
2. Neurons, Synapses and Weights documentation pages covering equations, model selection, units, initialization scaling, constraints and runnable examples. Neurons covers both COBA and CUBA versions of LIF, ALIF and AdEx, including their distinct input equations, compatible synapses and defaults. Its runnable example compares all three models separately under current and conductance input, with downloadable source, network diagrams and voltage/adaptation/spike plots for both families.

## [0.6.0] - 2026-10-06

### Added

1. Custom extension callbacks accept persistent `module:function` references and imported functions. Compiled graphs and training recipes save callback locations, units, state ports and optional exact distribution versions as authenticated JSON metadata. Bundle loading, graph simulation and training restore definitions automatically in fresh processes; Python source and pickled functions remain outside bundles. Updated Customisation with an external callback module and a load-only `--simulate` command, API documentation and downloadable examples.

### Fixed

1. Missing bundle paths now raise an explicit `FileNotFoundError` and produce a clear CLI error. Missing extension modules/callbacks, incompatible declared package versions and conflicting registrations fail before execution. Saving non-importable callbacks fails before creating bundle files; older name-only bundles retain their manual-registration requirement.

### Changed

1. Manual publishing and documentation workflows accept an explicit `skip_tests` option for releases requested without rerunning tests. Automatic workflows and manual runs retain tests by default.

## [0.5.0] - 2026-10-06

### Upgrade notes

1. This is a breaking release: the CLI and Python request API execute authored graphs only. Replace flag-built networks and config replay with `snnlab.lang` bundles, explicit input bindings and training recipes. `--executor graph` remains accepted and is optional; `--executor legacy` is rejected.
2. Import measurements from `snnlab.analysis`. Load graph checkpoint directories or native `GraphExecutor.state_dict()` files; historical COBANet matrices require an explicit conversion in the consuming application. Construct general `snnlab.viz.Recording` objects from named signals instead of using the removed snapshot loader.
3. Shared numerical kernels, dataset loading, Poisson image encoding, graph voltage-sampling compatibility and persisted graph/runtime/checkpoint schemas retain their supported contracts. Preserve historical run configurations and evidence unchanged.

### Removed

1. Removed legacy Config/COBANet execution, predefined network construction, training/inference/probe runners, mutable network-size globals and specialised run logging. The CLI and typed request API now support only graph execution; `--executor graph` remains optional. All commands require explicit bundle, input and output paths.
2. Removed the remaining 45 deprecated CLI switches, `dump-weights`, config replay/default restoration, bundle-to-COBANet translation and built-in legacy parameter interchange. Graph/PyTorch checkpoints, runtime continuation, interventions, recordings, training recipes and current numerical kernels remain supported.
3. Removed unused SimulationSpec/simulation.json authoring and execution, the legacy recording loader, `snnlab.sim.metrics`, specialised `compute_metrics`, legacy batch encoding, single-image loading and dataset-dependent network-size defaults. Import measurements from `snnlab.analysis`; keep using `load_dataset` and `encode_images_poisson` in graph applications.

### Validation

1. Preserved authenticated pre-removal numerical fixtures covering ten forward cases and a four-update optimizer trajectory; graph continuation/resume and independent Brian2 checks remain in the suite. Continuous voltages/logits use explicit `1e-6` absolute/relative tolerances across CPU architectures; forward spikes, conductances, parameters and same-platform continuation still require exact equality. Retired tests specific to removed execution features and corrected obsolete import paths in shared neuron-kernel tests.
2. Local validation passed 556 tests with one accelerator test skipped, lint, documentation type checking, root and GitHub Pages static builds, package builds and installed-wheel graph execution. Confirmed that all 12 retired modules are absent from the wheel.

## [0.4.0] - 2026-10-06

### Added

1. Added the flat `snnlab.analysis` API with 24 public functions for population activity, event intervals, rhythmicity, FFT/Welch spectra, irregularity, correlations, burst detection, cycle occupancy and conductance trajectories. Shared binning exposes rounding, alignment and incomplete-bin policies; new measurements return resolved estimator metadata and explicit undefined-result status. Existing rhythmicity calculations, population-count CV defaults and `snnlab.sim.metrics` imports remain compatible.

2. Added an Analysis guide, complete function reference and downloadable saved-activity example. Added known-signal tests for rates, sinusoidal power, correlations, trial boundaries and cycle opportunities, alongside historical-estimator comparisons.

### Removed

1. Removed `snnlab.lang.components` and its `ping` circuit helper. Author E/I circuits explicitly with `Network.population` and `Network.connect`; existing serialized bundles remain supported. Regression tests retain their circuit builder as a shared test fixture.

2. Removed the `snnlab.sim.accelerator_forward` validation helper and its dedicated tests.

## [0.3.0] - 2026-10-06

### Upgrade notes

1. This release includes the Pinglab migration APIs for batchwise encoding, epoch observations, checkpoint selection, spike replay, selective resets, decision segments and bounded inference retention. Refractory-duration unification, autapse policy and full expected-request/checkpoint authentication remain separate work.
2. Removed legacy CLI switches now fail explicitly, and retained deprecated switches emit a visible warning. See the [CLI reference](docs/content/docs/api/sim/cli.mdx) for the complete removal list; historical config replay and Python APIs retain their supported behavior.
3. `recording` / `--recording` now accept the new typed `RecordingSpec` / JSON selection contract; old recording profile strings remain unsupported. Dataset checkpoints produced under the older eager encoding protocol cannot resume under the new batch encoding protocol. Runtime-state schemas 1–3 remain readable; explicit continued measurement windows use schema version 4.

### Added

1. Added serialized `initialization_scaling="direct"` / `"fan_in_normalized"` controls to `Network.connect`, `Network.parameter` and `readouts.MeanVoltage`. Direct initialization preserves legacy readout weight scale without compensating initializer values or mutating tensors. Shared projection weights initialize once and reject conflicting policies. Older graphs retain normalized projection and direct operation defaults; checkpoint weights are loaded without rescaling. Updated API references and tutorial explanations.

2. Added opt-in `EpochObservations` and named `ObservationProbe` draws to graph epoch training: population rates in Hz, stored parameter L2 norms, output-population spike totals/silence/shares, and pre/post-clipping epoch gradient norms. Compact spike counting avoids additional dense observation histories; epoch evaluation preserves Python, NumPy, CPU and accelerator RNG streams. Audited checkpoints retain authenticated completed history and partial-epoch gradient totals, reject incompatible requests and do not fabricate missing past measurements. Updated request/result references and the Training tutorial.

3. Added explicit `CheckpointSelection` / `SelectionMetric` policies for completed-epoch graph-training checkpoints, including unweighted validation cross-entropy with accuracy tie-breaks. Selection evaluates the saved weights, records policy/scores/epoch/update and content-verified evaluation identity, and preserves the best candidate across checkpoint resume. Initial eligibility and exact ties are explicit; missing/non-finite metrics and incompatible resume requests fail. Added JSON CLI policy support, schema-version-4 epoch-selection checkpoints, request/result references and a Training tutorial example; the named legacy batch-loss default remains compatible.

4. Added typed `ExecutionSpec.interventions` (`DropSpikes`, `AddPoissonSpikes`, `ReplaySpikes`) and authenticated dense/sparse population-spike replay. Replay preserves ordinary neuron updates while replacing emitted spikes before downstream propagation, delay histories, recordings and readouts; absolute coverage supports exact continuation. Added pinned replay-file loading, CLI replay syntax, expected-intervention cache validation, bounded legacy/graph and continuation tests, and API/CLI documentation. The old `options["inference_interventions"]` and direct-forward intervention mappings now fail with migration guidance.

5. Dataset snapshot graph requests now automatically encode one batch at a time over verified temporary memory-mapped source arrays. Training gets fresh feature-scaled fixed/categorical Poisson draws; evaluation supports explicit repeatable validation draw sets. Versioned order/rate/spike streams and iteration coordinates reproduce resumed training and reject incompatible older protocols. Added automatic one-epoch/default-32-batch dataset behavior, batchwise inference with one model load, exact replay support, CLI categorical rates, memory/resume fixtures, and request/result/tutorial/CLI documentation.

6. Added typed absolute `BoundarySchedule`, selective `ResetVoltage` controls and online `DecisionSegments` to graph simulation/inference, with direct CLI JSON flags. Boundaries close the preceding segment before resetting the chosen leaky-integrator voltage and updating the next timestep; hidden, refractory, synaptic and delay state continue. Completed per-batch population totals and output-neuron spike counts appear in results and inference artifacts; compact partial counters and authenticated schedule metadata survive runtime save/load continuation. Reset/decision policy binds inference cache identity, with explicit mismatch checks and API/CLI documentation.

7. Graph inference now automatically accumulates supported time sums, means and final selections, including linear readouts and masks, when full histories are unnecessary. Added explicit absolute measurement windows with portable partial reduction state, selected-cell dense and sparse spike-event recordings, CPU recording sinks/NPZ blocks, and optional output retention. Dataset inference/evaluation uses online reductions and one model load; sinks can avoid concatenating output/final-state histories across samples. Recording/window policy binds cache identity, diagnostics preserve dynamics/RNG and autograd training retains its full history. Added equivalence, continuation, allocation-scaling, segment/reset, CLI and artifact fixtures plus API/result/CLI documentation.

8. Added a dedicated GraphExecutor API reference and simulator package guide covering direct PyTorch construction, parameters, constraints and continuation.

### Fixed

1. Newly authored graphs now use explicit leaky-integrator voltage sampling: `.voltage` remains post-reset membrane state and `.pre_reset_voltage` exposes the pre-reset integration sample. Masked and unmasked sum/mean/final operations consistently use the declared signal; `MeanVoltage` explicitly selects pre-reset voltage to preserve classifier behavior. Serialized `voltage_sampling="explicit"` distinguishes the corrected contract; older graphs without it retain historical reductions without rewriting checkpoints or artifacts. Added threshold-crossing, masks, gradients, continuation, diagnostics and legacy-compatibility tests, and documented phase and measurement-window semantics.

2. Documentation development now reloads page content, additions/deletions and navigation from current files, and resynchronizes changelog/example sources without restarting the server. Static documentation routes remain unchanged.

### Changed

1. Removed 34 legacy simulator CLI switches without experiment callers in the audited snnlab/Pinglab sources. Supplying a removed switch now produces an error; the CLI reference lists all removals. Marked 45 retained legacy switches deprecated in help; explicitly supplying them emits one visible `LegacyCLIWarning` per invocation listing the deprecated switches used. Shared and graph controls remain ordinary arguments; legacy execution defaults, Python APIs and historical config replay are preserved. CLI abbreviations are no longer accepted.

2. Removed the root release-helper scripts and separate migration/versioning documents. Release preparation is manual, with instructions in the README; CI and publishing read the package version directly.

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
