# Streaming deadlines and processing costs

Run from the repository root:

```sh
uv run python benchmarks/streaming/benchmark.py \
  --hidden-sizes 64 128 256 512 --windows 5000 --seeds 7 8 9 10 11
```

Results are saved directly into `benchmarks/streaming/results`, regardless of the
working directory: a flat list of files, overwritten on each new run. Generated
files from older cases are removed so changing the size list leaves no stale
measurements. The only saved files are `report.json` and `streaming.png`.
`--output` optionally selects a different directory with the same overwrite
behaviour; unrelated files there are preserved. One script builds each network,
warms and checks it against eager execution, then writes **one multipanel figure,
`streaming.png`**.
By default, five consecutive seeds starting at `--seed 7` are used (7–11).
`--seeds` supplies an explicit list of distinct nonnegative seeds. Seeds change
both initialized weights and Poisson input draws. Size order rotates between
seeds to reduce a fixed network-size/time-of-run confound. Runs execute serially
to avoid benchmark processes contending with one another.

Each network occupies one row with four panels, averaged across seeds:

1. Full callback, input preparation, prepared stream advance, output extraction,
   and measurement remainder: paired mean wall/thread CPU bars with SD bars.
2. Phases inside `stream.advance`: paired mean wall/thread CPU bars with SD bars. Persistent
   block sessions omit per-window state restoration and snapshots; fallback
   sessions include them.
3. Callback and release-to-completion timing: means within fixed 100 ms intervals,
   then means and SD bars across seeds. The 1 ms budget and worst individual
   completion remain visible; averages do not establish deadline compliance.
4. Deadline miss rates and callback overrun rates in fixed 100 ms intervals
   of scheduled release time. Overall totals remain in the title. Each interval
   uses its actual window count, including a partial final interval; empty
   intervals remain gaps. Rates are computed separately from each seed's original
   window timestamps, then combined as means and SD bars. Pooled miss totals
   and mean ± SD missed windows per seed remain in the title. All networks share
   the same percentage scale.

SD is the sample standard deviation (`ddof=1`) across seed-level measurements;
it is not divided by the square root of the number of seeds. The replicates are seeds, not
the thousands of windows. Each seed receives equal weight. Seed variation here
also includes ordinary host scheduling variation; five seeds cannot establish a
hard deadline guarantee. With only one seed, SD is unavailable.

The command above measures 5 seconds per seed/network combination: 100 seconds
of paced measurements in total. Compilation, numerical checks, unpaced diagnostics,
JSON output and plotting add wall time. `wall_time_seconds` records main-entry
through final plot generation, excluding Python/import startup and the final JSON
write.

The paced pass defaults to **one release every 1 ms, with a deadline 1 ms after
that release**. `--period-ms` sets both release spacing and deadline budget;
`--window-ms` independently sets the amount of model time advanced per callback.
Defaults therefore process ten 0.1 ms numerical steps in each 1 ms real-time slot.
One monotonic clock measures releases, starts and completed callbacks. Releases
stay anchored to one epoch: late callbacks catch up without dropping simulation
steps or resetting the deadline schedule.

The default `--spin-us 1000` polls the monotonic clock between callbacks at a
1 ms period, occupying approximately one full CPU thread to reduce wake-up jitter.
With a longer period it sleeps until 1000 µs before release, then polls.
`--spin-us 500` is a lower-CPU hybrid alternative; `--spin-us 0` restores sleep-only
waiting. All settings preserve the same absolute releases and deadlines. Reports
record the guard and measured caller-thread CPU percentage over the paced pass;
plot titles show the across-seed CPU mean. No process priority, affinity or global garbage
collector settings are changed. Smaller observed miss counts are not a guarantee
of zero future misses.

A **deadline miss** means completion after scheduled release + period. A
**callback budget overrun** means callback duration exceeds the period. A callback
can miss its deadline even when its computation fits the budget, because it
started late. The paced callback includes Poisson input generation/transfer,
`stream.advance`, host spike/event extraction, and device completion. Warmup and
compilation occur before measurement. Sleep/OS scheduling and measurement/storage
overhead can delay starts; these are observations on the measured host, not hard
real-time guarantees or ROS/sensor/actuator timings.

The two breakdown panels use separate warmed **unpaced** diagnostic passes.
Full callback/advance is an inclusive total; remaining phases sum to it. Device
fences and phase instrumentation affect timings. Thread CPU excludes other
workers and accelerator compute. The paced pass has no internal phase probes.

The workload is batch-one Poisson input → recurrent COBA LIF hidden layer → COBA
LIF readout, with random untrained AMPA weights stored as dense matrices containing
90% zeros. Defaults are 64 inputs, eight readouts, one CPU thread, FP32,
25 Hz input per channel, and ten-step compiled blocks. Input and eligible terminal readout projections are
batched within compiled blocks. Numerical checks require exact spikes and final
state agreement within 3e-4. Internal instrumentation is also checked against an
uninstrumented run.

Use `--device cpu|cuda|mps`, `--compile-mode default|eager`, `--block-steps`,
`--inputs`, `--readouts`, `--sparsity`, `--rate-hz`, and `--seed` to vary the workload.
MPS supports compiled blocks up to four steps; use `--block-steps 4` there.

`report.json` stores every seed's configuration, validation, phase means, timing
percentiles, deadline counts and per-window paced release/start/finish timestamps
relative to that seed's epoch. Per-network summaries store across-seed means and
SD, along with pooled miss counts. The report also records actual execution order
and whether all requested seeds completed. These timestamps are sufficient to
redraw timing and interval-rate panels; no separate sample files are written. Redraw without executing a simulation:

```sh
uv run python benchmarks/streaming/benchmark.py \
  --plot-only benchmarks/streaming/results
```

Only the current JSON report and PNG figure are kept in the results directory.
Operator-profiler and cProfile captures are not part of this benchmark.
