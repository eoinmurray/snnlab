# SHD native-training benchmark

A feedforward **700 → 128 COBA LIF → 128 COBA LIF → 20 leaky readout**
classifier, using the calibrated COBA settings from Pinglab's SHD experiment.
The default is **10 epochs, batch 256, seed 42, 1,000 timesteps at 1 ms**.
There are no inhibitory populations or recurrent connections.

## Implementation and protocol

1. `shd_recipe.py` declares the graph and `TrainSpec` with `snnlab.lang`.
   `benchmark.py` calls `snnlab.sim.train()` once. The native trainer
   owns initialization, batches, encoding seeds, forward/backward, Adam, constraints,
   validation, observations, timings and the final resumable checkpoint.
2. Public registered extensions preserve the source's centered readout initializer,
   fixed gain initializer, rate normalization, equal-class cross-entropy, Adam
   factory and noisy event encoder. Their small tensor functions are policies used
   by the native trainer; there is no custom PyTorch model or training loop.
   Keep this directory on `PYTHONPATH` when loading its graph/checkpoint elsewhere.
3. E membrane tau is 20 ms, capacitance 1 nF, leak 0.05 μS, rest/reset −65 mV,
   threshold −50 mV and refractory period zero. AMPA decay is 10 ms. Readout
   membrane tau is 20 ms and its projection decay is 10 ms.
4. Input/interlayer initializers are lower-clamped Gaussian parents with mean
   0.09 μS, SD 0.009 μS and 95%/90% initial zeros. Native fan-in scaling divides
   by source count and retained fraction. Nonnegative weights can become dense;
   these are not permanent connectivity masks. Readout weights have SD 0.1 and
   zero column means, with direct scaling.
5. Scores are temporal mean readout voltage, multiplied by fixed gain 20, then
   normalized by mean E2 rate with a 20 Hz reference and 1 Hz denominator floor.
   Objective: mean cross-entropy within each represented class, then across classes.
   Reported cross-entropy remains the ordinary example average.
6. Adam betas are 0.9/0.999, weight decay zero; learning rates are 5e-6 input,
   2e-5 interlayer and 0.01 readout. Gain is frozen. Fast-sigmoid surrogate slope
   is 100/15 per mV, scale 0.01, exclusive threshold; voltage-gradient damping is
   1. No clipping or rate regularization. Every utterance starts with reset state.
7. The checksum-verified official SHD training archive is split by per-class
   seeded permutation into 7,340 training / 816 validation utterances. The
   official test set is never loaded. This is an utterance holdout, not a speaker
   holdout. Events at/after 1 s are discarded; remaining events are floor-binned
   with binary OR. Insertion is 1.6 Hz/channel and occupied-event deletion
   probability 0.0008. Native indexed seeds vary training corruption by epoch
   while validation corruption stays fixed.
8. Native class-stratified ordering uses every utterance once per epoch; tails
   smaller than 20 are merged. Full data gives 28 batches of 256 plus 172:
   **29 updates/epoch, 290 updates total**. Online train metrics use pre-update
   batches; full validation runs initially and after every epoch. There is no
   extra training-set evaluation or best-checkpoint selection.
9. CUDA defaults to compiled neuron kernels; CPU defaults to eager. Projection
   arithmetic is fast, extension validation structural and dataset prefetch 1.
   Unused training snapshots are suppressed by the native trainer.

## Run locally

From the repository root after `uv sync --dev`:

```sh
uv run python benchmarks/shd/benchmark.py --device cuda:0 --output artifacts/shd/run-001
```

The cache defaults to `artifacts/shd-cache`. `--cache` changes it. Output must
be a new directory, preventing accidental replacement of checkpoints/results.
`--epochs` permits shorter pilots; `--compile-mode eager` disables compilation.

Download-free CPU plumbing check (40 synthetic training / 20 validation items,
64 steps, two epochs, batch 256; **not SHD learning or performance evidence**):

```sh
uv run python benchmarks/shd/benchmark.py --smoke --device cpu --output /tmp/shd-smoke-001
```

## Modal H100

```sh
uv run --with modal==1.6.1 python benchmarks/shd/modal_app.py
```

This Python launcher uses **`app.run(detach=True)`** and submits one worker
with `.spawn()`, returning a run ID, function-call ID and local `cloud.json`
submission ticket. **Once it prints `Submitted …`, your Mac can sleep or shut
down.** Source upload and submission must finish first. The worker owns training,
the optional diagnostic, report creation and the final volume commit; nothing
needs a local process to continue orchestrating work.

The equivalent explicit Modal CLI command is:

```sh
uv run --with modal==1.6.1 modal run --detach benchmarks/shd/modal_app.py
```

Keep `--detach` when using `modal run` directly; an attached Modal app still
follows the client's lifetime. See [Modal detached apps](https://modal.com/docs/guide/apps#ephemeral-apps).

Add `--smoke` for a small compiled CUDA check or `--epochs 1` for a one-epoch
pilot. Each launch uses a new run ID. Resources: four CPU cores, 16 GiB host RAM,
one H100 and a two-hour worker timeout. Logs remain in the Modal dashboard.
The cache, snapshot, bundle, checkpoint, reports and completion manifest persist
on the `snnlab-shd` volume. After successful completion, collect later with:

```sh
uv run --with modal==1.6.1 python benchmarks/shd/modal_app.py --mode collect --run-id shd-REPLACE_WITH_RUN_ID
```

Collection reads the volume directly and **does not start a GPU worker or rerun
training**. Small reports go to `benchmarks/shd/results/<run-id>/`, preserving the
submission ticket's call ID. A missing completion manifest is an error; check
whether the job is still running or failed in the dashboard. Worker exceptions
propagate and its `finally` block commits partial artifacts for inspection.
Raw traces, snapshots and checkpoints remain on the volume for explicit retrieval
with `modal volume get`. The new workload has not yet been timed on H100; the
15–20 minute budget remains provisional.

## Outputs and interpretation

1. `network.bundle/`: exact graph, training recipe and portable extension references.
2. `spikes.npz`: ragged event snapshot, labels and official training indices;
   only dense minibatches are materialized. `configuration.json` records split
   and source hashes, runtime versions, precision settings and device details.
3. `metrics.json`: native epoch learning metrics, parameter norms and execution
   protocol. `timing.json`: synchronized native setup, training, validation,
   checkpoint and reporting timings. Lazy compilation belongs to the first
   relevant phase; cold and warm epochs must be interpreted separately.
4. `final.checkpoint/`: native parameters, optimizer/RNG/data-order state after
   the last epoch. Resumption requires the exact snapshot and compatible recipe;
   moving files may require restoring the original recorded dataset path.
5. `summary.json`: completed updates, broader wall time, peak CUDA allocated and
   reserved memory. These memory peaks cover the whole training call, not just
   forward or backward. They are capacity measurements, not bandwidth counters.

This baseline measures ten epochs of learning plus validation and persistence.
Optional diagnostics below add sparsity, profiler and activity measurements. Device
DRAM bandwidth and hardware-counter utilization remain unavailable. Accuracy, activity, gradients
and full training outcomes need validation when optimizing execution later.

Settings copied from `experiments/exp124/recipe.py` and the native `train_condition`
in `compute.py`; the benchmark has no runtime dependency on Pinglab. It preserves
the native recipe, not the older explicit-loop initialization draw/encoding order.
Dataset: [SHD](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/),
[archive checksums](https://zenkelab.org/datasets/md5sums.txt).

## Local verification

1. Download-free acceptance tests plus native training-policy/timing regressions:
   `python -m pytest tests/sim/test_shd_benchmark.py tests/sim/test_training_policies.py tests/sim/test_timing.py -q`
   passed **82 tests**, with two accelerator-dependent skips on the CPU host.
2. A direct comparison against the source native COBA recipe matched every initial
   parameter and all outputs exactly on a seeded 1,000-step, two-example CPU probe.
3. The cached real SHD archive passed its published checksum. Preparation produced
   7,340/816 examples; one native CPU update with batch 256 and 1,000 steps completed.
4. The two-epoch synthetic smoke run changed all three learned matrices, left the
   gain unchanged and wrote a bundle that loads in a fresh Python process.
5. The launcher imports with Modal 1.6.1. Neither an H100 run nor the full ten-epoch
   learning trajectory has been executed for this new benchmark.


## Optional tracing

Tracing is **off by default**. Local and Modal launchers accept `--trace-mode`
(`off`, `monitor`, `profile`, `both`), `--trace-skip-updates` (default 1) and
`--trace-updates` (default 3). Local `--no-trace-detailed-labels` disables
per-timestep projection/neuron labels; the Modal Boolean option is
`--no-trace-detailed-labels` as well.

```sh
uv run python benchmarks/shd/benchmark.py --device cuda:0 --output artifacts/shd/profile-001 --trace-mode profile --trace-skip-updates 5 --trace-updates 3
uv run --with modal==1.6.1 python benchmarks/shd/modal_app.py --trace-mode both --trace-skip-updates 5 --trace-updates 3
```

Download-free combined-mode CPU check:

```sh
uv run python benchmarks/shd/benchmark.py --smoke --device cpu --output /tmp/shd-trace-001 --trace-mode both --trace-skip-updates 0 --trace-updates 1
```

1. The complete normal training run always executes first. Its metrics, timings,
   checkpoint and memory summary remain the baseline. Tracing then runs a **new
   native train call from the same seeded initialization**, with identical training
   batch/encoding policy, limited to skipped plus requested updates. It does not
   resume from the trained baseline checkpoint. Diagnostic validation, parameter
   observations, epoch callbacks and checkpoint persistence are disabled. No custom
   simulation/optimizer loop is introduced.
2. `monitor` collects detached spike/weight/gradient statistics, phase timings,
   boundary memory and process CPU measurements. `profile` collects PyTorch
   operator/allocation tables and a timeline. `both` combines them. Warmup updates
   execute but are outside the capture window. Short runs explicitly report a
   truncated or empty capture. The diagnostic is additional GPU work beyond the
   baseline timing budget, and its timings are instrumented.
3. `tracing/summary.json` and CSV tables describe measured, derived, estimated and
   unavailable quantities. `trace.json` is produced by profile modes. Do not call
   kernel coverage SM utilization or copy rates device DRAM bandwidth. Complete
   scopes are documented in [Tracing](../../docs/content/docs/api/sim/tracing.mdx).
4. Later collection downloads small diagnostic summaries/CSVs alongside baseline reports.
   Large raw traces remain on the volume; `cloud.json` gives the path. Retrieve
   explicitly with `modal volume get snnlab-shd runs/<run-id>/tracing/trace.json /tmp/shd-trace.json`.
   The final baseline checkpoint remains independent of diagnostic updates.

## Retained-run report

The docs report and four PNG/SVG/PDF figures use only `shd-59e3371bb384`.
Generate them without training:

```sh
uv run python benchmarks/shd/plot.py benchmarks/shd/results/shd-59e3371bb384
```

To regenerate the compact timeline excerpt, retrieve the raw trace to temporary
storage and add `--trace /tmp/shd-trace.json`. The extractor streams the file
and aggregates only actual CUDA kernel events, excluding annotation spans.
`report-data.json` and `presentation.json` bind metrics, trace excerpt and figures
with source hashes. The report's spike/gradient measurements cover diagnostic
batches 6–8, not the final trained checkpoint. `docs/scripts/sync-shd.ts` validates
and copies the retained figures/data for the site.

## Encoder device policy

`encode_events` accepts an optional `device` keyword. CPU execution retains the
existing NumPy `default_rng(seed)` stream and binning. CUDA/MPS transfer compact
batch event coordinates, bin on the device, and draw insertion/deletion noise
with a local seeded `torch.Generator` on that device. Original occupancy is
captured before either noise operation; deleted occupied bins are not reinserted.
The device stream is repeatable under the same backend/software policy, without
claiming NumPy-equivalent draws. Accelerator protocols record
`snnlab.encoder-device/v1`, preparation backend and PyTorch version; CPU encoder
protocols keep their previous identity. A checkpoint with another preparation
policy is rejected instead of silently switching its future inputs.
