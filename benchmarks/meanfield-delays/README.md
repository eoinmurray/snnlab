# Adaptive transmission-delay performance

Run from the repository root:

```sh
uv run python benchmarks/meanfield-delays/benchmark.py \
  --baseline-ref a4165349877a22325f18bf80570e163b6daea60d
```

The benchmark uses the four-population delayed E/I example: 14 states, 0.1 ms
internal delays, 2 ms cross-circuit delays, and a delivery-time coupling switch at
500 ms. Each run integrates 600 ms using DOP853, `rtol=1e-8`, `atol=1e-10`,
`max_step=0.1` ms and observations every 0.5 ms. Override `--repeats`, `--duration`
or `--output` when needed. A duration below 500 ms excludes the coupling switch.

1. The baseline integration and numerical RHS wrapper are loaded from the
   specified local Git revision. No checkout or source files are changed.
2. Both implementations integrate the same currently built model, initial state,
   constant prehistory and controls. Lazy symbolic compilation is excluded from
   timing, as are imports, model construction, plotting and output serialization.
3. Three repetitions alternate case order. The report records individual timings,
   medians, speedups, segment counts, environment and baseline revision.
4. Delayed trajectories are compared against the baseline at identical observation
   times, with `rtol=1e-9`, `atol=1e-10`. Maximum absolute differences are recorded
   per state with its native unit. The ODE comparison is checked separately.
5. The ODE cases explicitly omit delays. They measure computational overhead;
   they are different equations and are not scientific substitutes for the DDE.

Results are saved to [results/report.json](results/report.json). Ratios describe
this local host and workload; they are not a general speed guarantee. The saved
report's changed-file list records whether the measured implementation included
uncommitted edits. Reproduce implementation changes using the corresponding
source revision or patch in addition to the recorded baseline.

The recorded 2026-10-25 run reduced median delayed integration time from 31.96 s
to 4.75 s: a 6.73-fold speedup and an 85.1% reduction in elapsed time. Relative to
each implementation's ODE, the delay penalty fell from 14.18-fold to 2.27-fold.
Both delayed implementations retained 6,000 segments. The largest absolute
sampled-state difference was 1.96e-14 Hz. Timings varied materially across the
three repetitions; the report preserves every individual measurement.

Validation included all 77 mean-field tests, with analytic oscillatory feedback
and shared-delay source ordering checked across RK23, RK45, DOP853, LSODA, Radau
and BDF. These checks establish the tested cases, not a global DDE error bound.
