# snnTorch / SNNLab comparison

Three code files:

1. `benchmark.py`: matched models, MNIST training, verification and pass timings.
2. `modal_app.py`: concurrent Modal training, isolated profiling and collection.
3. `plot.py`: the two-panel Pinglab house-style figure.

Three local result files:

1. `results/results.json`: protocol, provenance, both learning histories and all
   available profiling samples. Its `cloud` section locates the retained data/checkpoints.
2. `results/training-curves.png`: inline/display figure.
3. `results/training-curves.pdf`: vector figure.

## Protocol

SNNLab 0.12.1 base (`b361c3d`) with unreleased training timing support, snnTorch
1.0.0 and PyTorch 2.10.0 on Modal L4. The exact uploaded source hashes are retained
in `results.json`; a frozen source snapshot was used for training and profiling.
Train on the same stratified 6,000 MNIST images; evaluate all 10,000 test images.
Seed 17, identical initial weights and minibatch order, 20 epochs, batch size 128,
784 → 1000 → 10 neurons, 25 steps, beta=0.95, alpha=0.9 exponential currents,
Adam lr=0.0005, and temporal membrane cross-entropy summed over steps.

Based on [snnTorch Tutorial 5](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html),
with nonzero synaptic filtering. SNNLab uses public exponential-current projections
and a registered current neuron matching the tutorial's delayed subtractive reset
and ATan surrogate. This is not its built-in CUBA LIF reset rule. Both use float32,
deterministic algorithms and disabled TF32. SNNLab uses fast projection mode,
structural validation, eager custom callbacks, and no unused result snapshots.
SNNLab compiles a `TrainSpec` and calls `snnlab.sim.execution.train()` once with
an `ExecutionSpec` and `ValidationSpec`; the library owns minibatches, gradients,
optimizer updates and checkpoints. Extension hooks preserve the matched temporal
loss, Adam optimizer, initialization and ATan surrogate. A zero-loss classification
objective reports spike-count accuracy without changing the training gradients.
Both libraries use the same `seed + epoch` shuffle and report loss/accuracy at
fixed weights after each completed epoch, as in the SNNLab training tutorial.
Partial minibatches are retained. Single-seed results establish no general winner.

Training uses two concurrent L4 workers. Profiling uses one L4 with randomized,
interleaved conditions, five warmups and 50 samples per library/operation for
initial weights and each final checkpoint. Each pass covers 128 images and 25
steps. Backward excludes its fresh forward/loss preparation; timings exclude
Adam, gradient clearing and transfers. All raw samples remain in `results.json`.

## Commands

Run from the repository root after `uv sync --dev`:

```sh
uv run python benchmarks/snntorch-comparison/benchmark.py --verify
uv run python benchmarks/snntorch-comparison/benchmark.py --smoke --output /tmp/snn-smoke
uv run --with modal==1.6.1 modal run --detach benchmarks/snntorch-comparison/modal_app.py --mode train
uv run --with modal==1.6.1 modal run --detach benchmarks/snntorch-comparison/modal_app.py --mode profile --run-id mnist-30fabde19a1a7aa1
uv run --with modal==1.6.1 modal run benchmarks/snntorch-comparison/modal_app.py --mode collect --run-id mnist-30fabde19a1a7aa1
uv run python benchmarks/snntorch-comparison/plot.py benchmarks/snntorch-comparison/results
```

`train` uploads the current source and returns a new run ID. `profile` and
`collect` reuse that ID. Data and optimizer checkpoints stay on the named Modal
volume `snnlab-snntorch-mnist-comparison`; they are not duplicated locally.
Collection writes one JSON and renders PNG/PDF using the checkout's Python.
Only explicit `train` or `profile` commands start GPU work. Synthetic smoke
figures are labeled and must not be treated as MNIST results.

## Measured results

Run `mnist-30fabde19a1a7aa1` used two concurrent L4 workers. Both completed 20 epochs
and 940 updates. Final test accuracy: snnTorch 85.20%, SNNLab 80.01%. Losses,
accuracies and update histories exactly match the preceding untimed run; timing
did not change those training outcomes. Cross-library training equivalence is
not established by this single-seed run.

[Training run](https://modal.com/apps/me-93863/main/ap-P6yReIInFoJqrVL5cvYYeK) and
[isolated profiling run](https://modal.com/apps/me-93863/main/ap-sQrEQUz0hAcM0eiTS7TaQT).

Both totals include setup, training, initial/completed-epoch train/test evaluations
and in-memory checkpoint construction. Data preparation, warmup, disk writes,
plotting and Modal volume commits are outside both timers. SNNLab uses
`training_timing=True`; synchronized phase timings are retained in each training
result's `training_timing` mapping. Training-loop time includes each library's
update bookkeeping and internal checkpoint-selection overhead. Epoch rows contain
training, train-evaluation and test durations; epoch 0 evaluations are included in
the phase totals. Workflow per epoch is an amortized total including setup and
initial evaluation; training per epoch averages the 20 measured training phases.

| Measurement | snnTorch | SNNLab | SNNLab speed advantage |
| --- | ---: | ---: | ---: |
| Total workflow | 162.39 s | 148.61 s | +9.3% |
| Workflow per epoch (total / 20) | 8.12 s | 7.43 s | +9.3% |
| Training updates | 59.94 s | 53.46 s | +12.1% |
| Training updates per epoch | 3.00 s | 2.67 s | +12.1% |
| Train/test evaluations, including initial | 102.42 s | 95.04 s | +7.8% |
| Isolated training forward | 27.60 ms | 21.86 ms | +26.2% |
| Isolated backward | 14.09 ms | 14.53 ms | -3.1% |
| Isolated inference forward | 25.75 ms | 23.64 ms | +8.9% |

Speed advantage is `100 * (snntorch_time / snnlab_time - 1)`; negative values mean
SNNLab is slower. Pass values are pooled medians of 150 measurements per
library/operation over the shared initial and both trained checkpoints. Each pass
covers 128 images and 25 steps. Backward excludes fresh forward/loss preparation.
These separate latency measurements do not sum to training-loop time. Identical
output spikes and finite outputs/gradients were verified at all three shared
weight states. Full raw timing samples remain in the single JSON result file.
Native resume checkpoints are retained on Modal beside the canonical profiling
checkpoint (`snnlab.native/` and `snnlab.pt`). Reprofiling requires the recorded
library source; collection does not require a matching local library snapshot.
