# Native fine-tuning benchmark

Run `uv run python benchmarks/native-finetuning/benchmark.py` for the tiny CPU
fixture. Opt into local CUDA with `--device cuda --compile`. No remote or paid
jobs are launched. Use `--epochs 4` to retain additional warmed epochs.

The four cases compare dense identity/explicit one-to-one inhibitory weights
and unsplit/accumulated effective batches, with identical fixed sensory and
inhibitory preparation policies on each device. Every case executes native
training, backward, clipping, AdamW, constraints, initial/epoch validation and
compact population-rate observations. Seven examples give effective batches
4/3; microbatch size two gives physical batches 2/2/2/1. Three fixed validation
draws are reused. This is a synthetic correctness and instrumentation fixture,
not a representative measurement of a 1,024-cell network.

`results/report.json` records device/library/precision policy, input shapes,
remainder coverage, complete call wall time, disjoint native timing phases,
per-epoch training/evaluation timings, requested/resolved compilation/replay and
fallback reasons, parameter bytes and CUDA peak allocated/reserved memory.
Peaks are process-level allocator statistics reset before each train call;
CPU device allocation peaks are unavailable and recorded as null. Setup excludes
caller graph authoring/source-weight preparation, while complete-call time
includes native snapshot opening, encoding, transfers and synchronization.
The first epoch includes numerical compilation/capture startup; later epoch
training phases measure warmed complete effective updates. Initial evaluation
can also warm kernels and is reported separately from training. PyTorch and
device caches remain shared across cases; this is not independent cold-process
benchmarking. Use fresh processes for cold-start comparisons.

Results make no general speed claim. In particular, accumulation and
per-presentation initial state currently fall back from CUDA replay; compiled
blocks and online-reduction fallbacks are reported independently. Float32
highest-precision reference tests compare identical materialized inputs, not
merely equal seeds across different device RNG implementations.
