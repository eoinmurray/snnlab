# Digits on Modal GPU — 20 October 2026

Full original workload: 20 epochs, seed 17, 1,437 training / 360 held-out digits,
16 timesteps, batch 128, hidden size 64, 240 updates. Both GPU conditions ran
serially on the same NVIDIA L4, PyTorch 2.10.0+cu128, with one Torch CPU thread.

| Condition | Full native train call | Median warm epoch | Final held-out accuracy |
| --- | ---: | ---: | ---: |
| GPU control | 30.242 s | 312.37 ms | 93.33% |
| GPU automatic defaults | 80.507 s | 22.56 ms | 94.17% |
| Previous Modal CPU automatic run | 3.896 s | 137.74 ms | 93.61% |

The automatic GPU run is **13.85× faster per warm epoch** than the GPU control,
but **2.66× slower for the complete call**. Compilation and capture dominate
this tiny workload: its first training epoch took 71.22 s, plus setup and initial
validation. The complete automatic GPU call took 20.67× the previous CPU call.
CPU remains the benchmark's default device.

Full native timings include compilation/capture, setup, initial and every-epoch
validation, and the final checkpoint write. They exclude imports, input preparation
and the results JSON write. Warm epoch medians cover training, validation and
reporting for epochs 2–20; they do not represent a separate warm full-call rerun.
Workflow times were 30.657 s / 80.845 s for GPU control / automatic; subprocess
times including imports were 38.596 s / 88.691 s.

The GPU control uses compiled single-step kernels, FP32/highest precision,
prefetch 0, replay off and automatic fused optimizer off. Automatic execution
selected BF16 autocast, high matmul precision, fused AdamW, prefetch 1 and CUDA
replay. All **240 updates replayed**, with **2 captures and no replay fallbacks**.
Training used compiled blocks of 16; validation retained the kernel loop because
of online reductions. Numerical precision changes the single-seed trajectory.

Both GPU reports have identical workload configuration and fixed dataset metadata
to the previous CPU report. The recorded SHA-256 hashes match every current
library Python source file. The three local digits tests and Ruff passed.
This is one sequential pair, with potentially shared compiler caches; it is not
an independent clean-cache or repeated statistical measurement.

[Exact summary and execution paths](summary.json), [raw reports and logs](scikit-digits/),
and [Modal run](https://modal.com/apps/me-93863/main/ap-U0uJkJY44TcV9u0cDfGril).
Bundles and checkpoints remain on the `snnlab-automatic-defaults` Modal volume.

Reproduce with a fresh run ID:

```sh
uvx --from modal modal run --detach benchmarks/execution/modal_defaults.py --suite digits-gpu --run-id UNIQUE_RUN_ID
```
