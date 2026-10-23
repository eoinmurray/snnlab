# Streaming timing bar charts

Run from the repository root:

```sh
uv run python benchmarks/streaming/benchmark.py \
  --hidden-sizes 64 128 256 512 --windows 256 \
  --output artifacts/streaming/timing-bars
```

The output directory must be new. One script builds the network, warms and checks
it against eager execution, measures it, and writes two bar charts:

1. `breakdown.png`: full callback, input preparation, prepared stream advance,
   output extraction, and measurement remainder.
2. `advance-internals.png`: phases inside `stream.advance`, including validation,
   state restoration, compiled numerical blocks, output assembly, and state snapshots.

Both show arithmetic means in microseconds, with paired elapsed wall-time and
calling-thread CPU bars. Full callback/advance is an inclusive total; the remaining
phases sum to it. The two charts use separate warmed, **unpaced** passes. Device
fences and instrumentation affect timings; these are diagnostics, not deadline tests.
Thread CPU excludes other workers and accelerator compute.

The workload is batch-one Poisson input → recurrent COBA LIF hidden layer → COBA
LIF readout, with random untrained AMPA weights stored as dense matrices containing
90% zeros. Defaults are 64 inputs, eight readouts, one CPU thread, FP32, 100 Hz input,
**1 ms windows with ten 0.1 ms numerical steps**, and ten-step compiled blocks.
Input and eligible terminal readout projections are batched within compiled blocks.
Numerical checks require exact spikes and final state agreement within 3e-4.
Internal instrumentation is also checked against an uninstrumented run.

Use `--device cpu|cuda|mps`, `--compile-mode default|eager`, `--block-steps`,
`--inputs`, `--readouts`, `--sparsity`, `--rate-hz`, and `--seed` to vary the workload.
MPS supports compiled blocks up to four steps; use `--block-steps 4` there.
Compilation and numerical validation occur outside the recorded measurements.

`report.json` stores configuration, validation and phase means; compressed NumPy
files retain per-window wall/thread-CPU measurements. Redraw the charts with:

```sh
uv run python benchmarks/streaming/benchmark.py \
  --plot-only artifacts/streaming/timing-bars
```

Previous reports and figures under `artifacts/` are preserved. The old deadline,
operator-profiler and cProfile command-line options have been removed.
