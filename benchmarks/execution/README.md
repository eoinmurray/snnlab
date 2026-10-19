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

Block compilation remains opt-in through `compile_block_steps=16` or CLI
`--compile-block-steps 16`; the default is 1. Opt-in training checkpoint protocols bind the block size and compile mode; default protocols retain their historical identity. Unsupported callback, recording,
online-reduction, intervention, reset/decision, tracing and partial-parameter
autograd cases report their reference-loop fallback. CUDA Graph replay is available only as the standalone experiment described below.
Retained measurements guide explicit tuning; they do not change GPU defaults.

MPS blocks above four timesteps currently select the kernel loop with an explicit fallback reason: larger Inductor training graphs exceeded Metal buffer bindings during local PyTorch 2.14.1 validation. Four-step MPS blocks remain opt-in; this is a conservative validated limit, not a promise for arbitrary graph sizes. CPU/CUDA requests retain the 1-64 range.

## Local validation, 19 October 2026

[Summary and raw samples](results/summary.json) retain 46 matched numerical gates
and 89 completed timing cases on an Apple M5 with 32 GiB RAM, PyTorch 2.14.1 and
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
Cold costs can dominate short runs, so the default remains one step.

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
native training orchestration. It does not add a public compile mode.

## CUDA Graph experiment, 19 October 2026

Modal H100, PyTorch 2.10.0/CUDA 12.8, float32 with highest matmul precision:
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
within declared tolerances, with no full-training accuracy claim. Defaults remain
highest matmul precision, default Adam and no autocast.

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
checks: 1,167 local passes, 71 skips, and 16 failures independently reproduced on
the clean 0.20.0 baseline. Remote regression runs passed 155, 12 and 35 checks
respectively; these sets overlap and are not additive coverage totals.
