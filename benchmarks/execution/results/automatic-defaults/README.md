# Automatic execution defaults — Modal benchmarks

Full original workloads, 20 October 2026. Status: **complete**.

A subsequent [digits GPU rerun](digits-gpu-20261020/README.md) on L4 measured
80.51 s for automatic execution including compilation/capture, versus 30.24 s
for its GPU control. Warm epochs improved from 312.37 ms to 22.56 ms; the
previous CPU full call below remains faster for this small workload.

| Benchmark | Hardware | Before native call | Automatic native call | Speedup | Held-out accuracy before / after |
| --- | --- | ---: | ---: | ---: | ---: |
| scikit-digits | Modal CPU, 1 Torch thread | 4.30 s | 3.90 s | 1.10× | 93.61% / 93.61% |
| snntorch-comparison | NVIDIA L4 | 166.23 s | 42.30 s | 3.93× | 80.01% / 85.64% |
| shd | NVIDIA H100 80GB HBM3 | 1107.88 s | 518.63 s | 2.14× | 47.30% / 46.32% |

The small CPU difference should be treated as timing variation: digits retains its
eager CPU recipe, so CUDA optimizations do not apply. All pairs preserve the
same samples, batch sizes, update counts, losses, evaluation schedule and seed.

The fresh snnTorch L4 run took **169.09 s**, reaching **85.20%**. Automatic SNNLab is **4.00×** faster in the same timing scope.

## Accuracy and memory

Automatic precision changed the learning trajectories: MNIST finished at 85.64% versus 80.01% in the FP32 control; SHD finished at 46.32% versus 47.30%, a 0.98 percentage-point decrease. These are the original single seeds. `precision="highest"` opts out of automatic BF16/TF32 while leaving other controls available.

SHD peak allocated CUDA memory was 6.20 GiB before and 9.14 GiB after. Peak allocator reservation was 62.28 / 52.96 GiB respectively. Reserved memory is a caching-pool measure, not live tensor memory. MNIST peaked at about 1.45 GiB in both conditions.

## Archived references

The earlier saved MNIST calls took 148.61 s (SNNLab) and 162.39 s (snnTorch) on L4. The earlier SHD native call took 766.12 s (12m46s) on H100; the new automatic call is 518.63 s (8m39s), about 1.48× faster against that historical measurement. Those runs have older sources/hosts and, for SHD, a different encoder RNG backend. The paired table above is the controlled comparison of the new defaults. The older digits 1.34 s figure is from macOS arm64 and is not comparable to the Modal CPU.

## Scope and policies

Current library with blocks=1, precision=highest, CUDA replay off, automatic fused optimizer off, prefetch=0 unless benchmark explicitly uses 1; MNIST native compilation off. This isolates the new defaults while retaining previously implemented GPU encoding and kernel improvements.

Automatic public controls, preserving benchmark-specific workload/evaluation settings. Digits deliberately stays eager on CPU.

MNIST uses 20 epochs, 6,000 train / 10,000 test images, 25 timesteps, batch 128,
1,000 hidden cells and 940 updates. Initial and every-epoch fixed-weight train
and test evaluation remain enabled. Its wall timer now starts before train().

SHD uses 10 epochs, 7,340 train / 816 validation examples, 1,000 timesteps,
batch 256, the original two-layer COBA graph and 290 updates. The custom equal-class
objective, rate normalization, Adam factory, stratified batches, initial/epoch
validation and parameter observations are preserved.

Automatic execution enables supported BF16 autocast/TF32, fused Adam, prefetch
and CUDA replay. Compiled blocks are selected automatically when supported;
these two scientific GPU recipes retain their recorded block fallbacks. Replay
can still capture their approved tensor callbacks. Raw metrics record actual
paths, capture counts and fallback reasons.

## Interpretation

- One before/after pair per workload; small CPU differences are timing variation, not evidence of an accelerator benefit.
- Conditions run serially on the same worker. Compiler caches can be reused; these are not independent clean-cache compilation measurements.
- Native timers include capture warmup/setup and required evaluations. CPU digits/SHD include requested checkpoint disk writes; MNIST excludes final disk writes as in its historical protocol. Process totals include imports/verification/warmup and possibly downloads.
- BF16/TF32 alter numerical precision and learning trajectories. Accuracy is reported for the single original seed.
- Historical measurements use older sources; paired current measurements isolate this default change. Historical SHD also used the old encoder RNG backend.

## Artifacts and reproduction

[Summary JSON](summary.json) contains exact timings, phases, accuracy, hardware,
configuration, update counts and execution policies. [Verification](verification.json)
and compressed logs retain local/GPU checks and known baseline failures. Raw
per-condition reports live under the run IDs listed in the summary. Checkpoints
and prepared datasets remain on the `snnlab-automatic-defaults` Modal volume.

```sh
uvx --from modal modal run --detach benchmarks/execution/modal_defaults.py --suite full
uvx --from modal python - <<'PY2'
import runpy
m = runpy.run_path('benchmarks/execution/modal_defaults.py')
m['collect']('RUN_ID_FROM_TICKET')
PY2
```

The failed early MNIST launcher under `defaults-full-final` is retained. Its
baseline wrapper lost its public signature; functools.wraps fixed the harness
before the completed `defaults-mnist-final` run. The CUDA validation run passed
all tests; its subsequent smoke encountered the same superseded harness issue.
No failed run contributes a successful timing above.
