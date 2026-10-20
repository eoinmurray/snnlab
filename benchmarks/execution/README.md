# End-to-end execution benchmark

From the repository root:

```sh
OMP_NUM_THREADS=1 uv run python benchmarks/execution/benchmark.py \
  --devices cpu mps --steps 33 --batch-sizes 8 --block-sizes 1 16 32 64 \
  --profile --output artifacts/execution/local.json
```

Use `--devices cuda --steps 128 1000 --batch-sizes 8 32 256 --channels 700
--cells 128` on a CUDA host for SHD-shaped synthetic inputs. Add `--coba` for
conductance dynamics. These are synthetic execution checks, not SHD accuracy
measurements. Large block sizes can require substantial compilation time and
memory. Run conditions serially on an otherwise idle device. Each validation/timing case has a 180-second POSIX wall-time limit (`--case-timeout`, 0 disables); timed-out cases are recorded under `failures` and are never reported as passing numerical gates or successful timings. Numerical mismatches and other errors still fail the run.

Each case keeps its model, input features, seeds, batch order and optimizer
settings fixed. It includes a final partial batch, delayed recurrent feedback,
autapse exclusions, constraints and a readout. An eager path, compiled individual
kernels and opt-in compiled network blocks are measured separately. CUDA/MPS
cases compare CPU encoding plus transfer with device encoding. The latter uses
a different, explicitly recorded RNG stream; an independent gate uses identical
spikes/weights for eager/optimized output, gradient, parameter and Adam-moment
comparisons. It fails the run on a mismatch. Float32 absolute/relative tolerances
are `3e-4`; optional reduced precision uses `5e-2` and establishes no task-accuracy
claim. Dtype preservation, full trajectories and continuation have separate
regression coverage in `tests/sim/test_execution_dtypes.py` and
`tests/sim/test_network_blocks.py`.

Provider construction, validation, model setup and device movement have a
separate `setup_s`. `cold_epoch_s` is the first measured epoch for that model,
including lazy compilation and optimizer initialization. It can reuse the
machine's compiler cache and kernels compiled by the numerical gate; it is not
a clean-cache compiler measurement. `warm_epoch_s` contains raw completed-work
wall times, fenced only at epoch boundaries. It includes encoding, transfers,
forward, loss/backward, optimizer, constraints and requested recording.
`numerical_gates[].validation_and_first_execution_s` separately includes matched
validation and the first eager/compiled forward/backward/Adam execution before
those timing cases. Suspend/resume or competing workloads invalidate wall-time
comparisons; rerun those samples on an awake, idle host.

`--mode infer` omits backward and updates. The benchmark deliberately uses dense
readout histories to compare the same reduction across paths; public inference
may instead select online reductions and report a block fallback. `--record`
records voltage and sparse spike events to a counting host sink, including final
partial blocks. This exercises transfers and sink invocation, not disk writing.
Recording selects the reference loop; it never disappears to improve throughput.
`--prefetch 2` measures bounded preparation/transfer lookahead.

`--profile` warms the diagnostic shape before running a separate bounded diagnostic after timing: at most one batch
and 128 simulation steps by default (`--profile-batches`, `--profile-steps`).
Encoding still prepares the full presentation; both lengths appear in the
profile scope. This prevents a one-second eager trace from retaining millions
of events and dominating host memory. JSON retains operator
counts, scalar reads, copy calls, device kernel event counts where available,
arguments, compiled graph, source hashes, Git state, software/hardware, dtype and the selected/fallback
execution path. Counts overlap and are not a wall-time breakdown. CPU RSS is the
process-lifetime high-water mark (units recorded); CUDA reports allocator peak
bytes, including cold execution, and MPS only current allocator bytes. CUDA profiles retain Kineto-reported copy bytes and host wait counts. Missing
exported copy rows do not establish zero transfer. Raw pinned-host allocator
counters are retained when the API exists; PyTorch 2.10.0 exported inconsistent
values (including negative freed counters), so those peaks are unreliable. MPS
device kernel counts and device DRAM bandwidth require an external profiler. Numerical gates hold their own model
temporarily; process RSS can reflect that earlier allocation.

Library execution now defaults to `compile_block_steps="auto"` (CUDA 16, MPS 4,
CPU 1), one-batch prefetch, supported CUDA BF16 autocast/TF32, automatic fused
Adam/AdamW and bounded CUDA replay. Unsupported controls retain their documented
fallbacks. Recording sink granularity remains explicit. Full-update replay is
still a standalone experiment; native replay captures forward/loss/backward.
The controlled synthetic pipeline below deliberately bypasses the library's
precision/replay wrapper so `--autocast`, `--matmul-precision` and optimizer
arguments remain independent experimental controls. Its defaults retain FP32
highest matmul precision and ordinary Adam. Native workload reruns exercise the
public defaults instead.

See [automatic-default benchmark comparisons](results/automatic-defaults/README.md)
for the full digits, MNIST/snnTorch and SHD Modal runs, accuracy and scope.

## Initial local validation, 19 October 2026

[Summary and raw samples](results/summary.json) retain 46 matched numerical gates
and 89 completed timing cases before the compiler precision correction below, on an Apple M5 with 32 GiB RAM, PyTorch 2.14.1 and
one CPU thread. Compressed reports preserve arguments/graphs where available,
source hashes, software, raw warm samples, startup costs and measurement limits.
The larger synthetic circuit has 700 input channels, 128 COBA cells, delayed
recurrence and a readout. Each presentation lasts 1,000 ms with 1 ms steps and
includes a final partial batch. Warm throughput includes preparation, transfers,
forward/backward, Adam and constraints:

| Device | Batch | Compiled kernel loop (samples/s) | Four-step blocks (samples/s) |
| --- | --- | --- | --- |
| CPU | 8 | 35.2 | 97.9 |
| MPS | 8 | 18.3 | 42.0 |
| CPU | 32 | 104.3 | 215.3 |
| MPS | 32 | 81.7 | 165.6 |

CPU remains faster than MPS in these cases; larger batches improve MPS throughput.
Individual compiled kernels regress against eager execution on some short CPU
cases. A tiny 1,000-step/64-block CPU timing case exceeded its 60-second deadline
and has no throughput result. Larger MPS blocks use the documented fallback.
Cold costs dominated some short runs in this historical snapshot; the current automatic policy is described above.

Separate reports exercise warmed profiler scopes, inference recording with
prefetch and the native SHD encoder/trainer on MPS (two synthetic smoke updates,
not SHD accuracy). Earlier bounded-profile counts can include shape compilation;
use `warmed-profile.json.gz` for dispatch comparisons. The source inventories
describe each measured revision; later persistence/device-contract fixes have
separate regression coverage. CUDA validation and replay results are reported below.

## Modal H100 runs

The detached launcher snapshots local sources and tests, runs each condition
serially on its assigned H100, and commits reports/logs after each job to the
`snnlab-execution` volume:

```sh
uvx --from modal modal run --detach benchmarks/execution/modal_app.py
# Use the run_id printed in the local ticket under artifacts/execution/.
uvx --from modal modal run benchmarks/execution/modal_app.py \
  --mode collect --run-id cuda-EXAMPLE
```

Collection reads persisted files without invoking a GPU worker and can retrieve
completed jobs while later jobs are running. `--suite diagnostics` runs boundary
regressions, warmed transfer/wait profiling, CPU comparisons and a native SHD
smoke. `--suite graphs` limits a run to the replay experiment. The full suite
covers short and one-second training, multiple block/batch sizes, partial
batches, recording/prefetch, float16 and bfloat16. Failures and timeouts remain in
the manifest and raw reports. The image pins PyTorch 2.10.0, the supported minimum;
local MPS measurements use the separately recorded local version.

`cuda_graph.py` is a standalone experiment using the simulator's compiled blocks,
a static device input buffer, capturable Adam and built-in constraints. It
checks outputs, gradients, parameters and Adam moments/step across three updates
with changing inputs, resetting simulation state for every presentation. Capture
and compilation/warmup costs are separate. Replay timing includes the device
input copy and complete updates from resident inputs; encoding and host transfers
are excluded and remain in the pipeline benchmark. Capture needs a new graph for
each shape and does not cover callbacks, recording, continuation or arbitrary
native training orchestration. It does not add a public compile mode; native replay has a separate `cuda_graphs` policy.

## Initial CUDA Graph experiment, 19 October 2026

Before the compiler precision correction: Modal H100, PyTorch 2.10.0/CUDA 12.8, float32 with highest matmul precision:
700 inputs, 128 COBA cells, delayed recurrence, readout, 16-step compiled blocks.
Each shape passed three output/gradient/parameter/Adam-state comparisons with
changing resident inputs at `atol=rtol=3e-4`. The reference uses the same compiled
blocks and capturable Adam. Separate pipeline gates compare those blocks to eager.

| Steps | Batch | Compiled loop (samples/s) | Replay (samples/s) | Compilation/warmup (s) | Capture (s) |
| --- | --- | --- | --- | --- | --- |
| 33 | 8 | 967.3 | 1,315.4 | 48.31 | 0.015 |
| 128 | 32 | 1,300.4 | 4,368.6 | 43.18 | 0.051 |
| 1,000 | 32 | 181.4 | 558.0 | 16.18 | 0.488 |

Timing includes input-buffer device copies, fresh presentation state, forward,
backward, capturable Adam and built-in constraints. Encoding and host transfers
are excluded. The warmup costs can reuse compiler caches populated earlier in
the same container and are not clean-cache compilation measurements. Capture is
shape-specific, with no recording, custom callbacks or runtime continuation.
These synthetic results support further integration work; they do not establish
task accuracy or a general native-training replay mode.

## CUDA precision and profiling scopes

`--matmul-precision high`, `--adam-execution fused` and CUDA
`--autocast bfloat16` are explicit benchmark choices. Matmul/optimizer gates
compare against eager highest-precision matmul and default Adam at `3e-4`.
Autocast keeps float32 parameters, runtime state and Adam state; its comparison
against the float32 reference uses `5e-2`. Direct `--dtype float16`/`bfloat16`
instead converts the model and state. These checks establish numerical execution
within declared tolerances, with no full-training accuracy claim. This isolated benchmark retains highest matmul precision, default Adam and no
autocast; the library public defaults are automatic as described above.

`--suite profile` repeats the corrected diagnostic window. Earlier profiles
prepared an additional batch before stopping; final profiles limit requests before
encoding/prefetch. The final recording diagnostic observes one 128-step, batch-8,
32-channel input transfer: 131,072 bytes for CPU encoding, or 1,024 compact feature
bytes for CUDA encoding. Both deliver eight pinned device-to-host recording chunks
(262,144 bytes), including all requested voltage/spike samples. Cached cell indices
remove 256 per-step host-to-device index uploads from this case. Missing profiler
copy rows in other samples remain missing, rather than being counted as zero bytes.

Modal runs allocate four CPUs; the pipeline reports observed four Torch threads.
The CPU comparison reports `x86_64` without an identified processor model. It is
not a matched-hardware comparison with the local M5. A separate native SHD smoke
uses one thread and passes two synthetic updates, parameter verification and
checkpoint writing on CUDA; it is not a SHD accuracy benchmark.

`--suite trajectory` checks current- and conductance-based one-second trajectories,
including exact spikes, voltage/synaptic histories, delayed state and a split
continuation at step 438. Both current- and conductance-based CUDA cases passed
for `[1000, 8, 700]` inputs and 128 cells with 16-step blocks (`3e-4` for
non-spike tensors; exact spike equality). `trajectory.py` also accepts `--device cpu` for local validation.

[CUDA reports and raw samples](results/cuda-summary.json) retain per-run source
hashes, settings, hardware/software, failures, raw warm samples and startup costs.
[Verification](results/verification.json) retains simulator and documentation
checks: 1,183 local passes, 73 skips, and 16 failures independently reproduced on
the clean current 0.21.0 baseline. The final PyTorch 2.10 CPU/CUDA regression run
passed 174 checks with 36 skips, including the reduced-precision CPU fallback and
public keyword API. Earlier runs passed 155, 12 and 35 checks respectively; these
sets overlap and are not additive coverage totals.

## Compiler precision correction

The initial direct-bfloat16 run rejected gradients outside the declared `5e-2`
tolerance: cast elision inside compiled kernels changed recurrent gradients by
more than 30% on CPU and CUDA. Preserving intermediate dtype casts fixes the
individual-kernel and network-block reproductions. Prepared kernels now request
Inductor's `emulate_precision_casts=True`; forward metrics record
`compiler_precision_policy="snnlab.eager-precision-casts/v1"`. Model and state
dtypes stay unchanged. A value-preserving absolute-value round trip also avoids
an ambiguous Metal bfloat16 overload. Actual CPU/MPS/CUDA regressions cover this
policy. PyTorch 2.10 reduced-precision CPU kernels use eager tensor graphs: its
vector backend rejects these casts, and scalar code generation still changes
gradients. Forward metrics expose `compiler_fallback`, and block requests report
the reference-loop fallback. Float32/float64 CPU compilation, CUDA/MPS code
generation and newer local CPU compilation retain Inductor. The original
rejected probes and intermediate compiler failures remain in the raw reports.

The public keyword execution API and restored epoch-zero validation defaults from
0.21.0/0.20.1 are retained. Final checks use those interfaces and settings; earlier
reports retain their own source hashes and versions.

## Final precision-preserving performance

The final performance run requested an H100; Modal supplied an **NVIDIA H200**.
The reported device identity takes precedence over the requested allocation.
On PyTorch 2.10.0/CUDA 12.8, four observed Torch threads, float32/highest matmul
precision, 700 input channels, 128 COBA cells, 1,000 timesteps and batch 256:

| Execution | CPU input preparation (samples/s) | CUDA input preparation (samples/s) |
| --- | --- | --- |
| Eager loop | 81.5 | 122.7 |
| Compiled kernel loop | 102.1 | 174.1 |
| Compiled 16-step blocks | 197.7 | 917.9 |

The **5.27x** compiled-block gain compares the two CUDA-preparation paths and
includes encoding, transfers, forward/backward, Adam, constraints and a final
partial batch. It uses the current cast-preserving policy; earlier measurements
retain their own source revisions and must not be treated as current throughput.
The first measured block epoch took 8.38 seconds with CUDA preparation; numerical
gates ran before that epoch, so it is not a clean-cache compile time.

A separate final H200 CUDA Graph experiment (1,000 steps, batch 32) passed three
changed-input update comparisons, including gradients and optimizer state.
Resident-input throughput was 144.9 samples/s for the compiled block loop and
568.0 for replay, with 60.97 seconds of compilation/warmup and 0.456 seconds of
capture. Encoding and host transfers are excluded. Replay remains a standalone
experiment. Final reports are in `cuda-370dfb99adca`; the directory name
`results/cuda-h100/` is historical and each report records the actual hardware.

## Automatic native training replay

Native `train(..., cuda_graphs="auto")` now captures compatible repeated float32
CUDA forward/loss/backward work by default. `cuda_graphs="off"` and CLI
`--cuda-graphs off` disable it. Optimizer execution, clipping, constraints,
reporting, evaluation and checkpointing stay in the existing loop, preserving
AdamW configuration and checkpoint compatibility. A two-shape cache handles
repeated full/partial batches; one-off or unsupported shapes run normally.
CPU/MPS, unmarked custom numerical callbacks and tracing also use that path.
Approved `capture_safe=True` callbacks and supported BF16 autocast can replay.
Metrics report captured shapes, replayed/normal updates and reasons for fallback.

This native scope differs from the older full-update experiment above. Its timing
must be measured separately; the historical replay speedups are not native-trainer
throughput claims. Startup includes compiler warmup and capture, and can outweigh
replay savings in short runs. `--suite native-replay` validates the public training
path against `cuda_graphs="off"`, including changed batches/targets, partial
batches, delayed recurrence, clipping, regularizers, validation/test and resume.

### Native replay results, 20 October 2026

On an NVIDIA H100, PyTorch 2.10.0, float32, a 64-input/32-cell recurrent CUBA
circuit ran 128-step presentations with batch size 32 and a final batch of one.
Each timed public `train` call includes input resolution/transfers, forward/loss/
backward, unchanged AdamW and constraints, default online epoch metrics, capture
setup and mandatory final checkpoint construction. There are no held-out
validation/test splits or observations in this timing.

| Scheduled updates | Replay off: full call (s) | Auto: full call (s) | Off / auto throughput (samples/s) |
| --- | --- | --- | --- |
| 30 (10 epochs) | 4.84 | 5.51 | 134.4 / 117.9 |
| 300 (100 epochs) | 14.69 | 9.12 | 442.5 / 712.4 |

These are compiler-warm calls, each rebuilding its own two-shape capture cache.
The longer call improved by **1.61x**, including capture setup; the short call
regressed by about 14% in elapsed time. Both checked outputs, gradients, parameters
and AdamW state against replay off at `atol=rtol=3e-4`. Each pair followed separate
first-call measurements. The longer run's first off call took 77.78 seconds,
including compilation; the following first auto call reused those compiler
artifacts and must not be compared as an independent clean-cache startup.

The final CUDA regression suite passed 23 tests with one MPS skip, including dense
diagnostics, unused trainable leaves, partial batches, RNG preservation and
checkpoint resume. The full local suite passed 1,188 tests with 80 skips; all 16
failures reproduce on clean 0.21.0. [Native replay validation and raw reports](results/native-replay-summary.json)
retain these results, earlier corrected test-fixture failures and source hashes.
