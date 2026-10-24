# Scikit-learn digits / conductance LIF

A small training benchmark using the 1,797 bundled 8×8 handwritten digits
from `sklearn.datasets.load_digits`. The default runs eagerly on CPU and needs
no downloads or compilation warmup. CUDA runs use the automatic compiled
execution defaults.

Run from the repository root after `uv sync --dev`:

```sh
uv run python benchmarks/scikit-digits/benchmark.py
uv run python benchmarks/scikit-digits/benchmark.py --smoke
uv run python benchmarks/scikit-digits/benchmark.py --epochs 30 --output /tmp/digits-run
uv run python benchmarks/scikit-digits/benchmark.py --device cuda:0
```

The default is 20 epochs, batch size 128, one CPU thread, seed 17, and a fixed
stratified 80/20 split (1,437 training / 360 held-out digits). Divide pixel values
by 16, then encode each image once as 16 binary Bernoulli spike steps at 1 ms,
with a maximum rate of 1,000 Hz. Training and evaluation use separate seeded
spike draws; each split replays its fixed spikes across epochs. `--smoke` uses
100 training / 40 held-out real digits, 2 epochs and 8 steps.

The network is 64 inputs → 64 built-in `lang.COBA_LIF` neurons → 10 linear class
scores. Inputs drive nonnegative AMPA conductances (2 ms decay, weights in µS).
LIF cells use 1 nF capacitance, 0.05 µS leak, −65 mV rest/reset, −50 mV threshold
and no refractory hold. The readout linearly projects hidden spike counts;
the classifier itself is non-spiking. Both conductance weights and readout
weights learn through `snnlab.sim.train`, using cross-entropy, a fast-sigmoid
surrogate, AdamW without weight decay, and gradient clipping at 1.0. Learning
rates are 0.0003 for conductances and 0.01 for the readout.

Training metrics are online minibatch measurements. Held-out metrics are
fixed-weight evaluations before training and after every epoch; the final epoch
is reported, with no best-epoch checkpoint selection. All training samples,
including the final partial batch, are retained. This is a quick single-seed
sanity/performance benchmark, with no independent generalization study.

Each run replaces `results/` beside `benchmark.py`, including old checkpoints.
`--output PATH` replaces that directory instead. It contains:

1. `results.json`: configuration, split indices, environment, learning metrics and timings.
2. `network.bundle/`: portable graph and training recipe.
3. `final.checkpoint/`: final resumable native checkpoint.

`native_train_s` covers the native trainer, including setup, initial and epoch
evaluations, and checkpoint writing. `workflow_s` additionally covers loading,
encoding and graph/bundle creation. Both exclude Python imports and the final
JSON write. Timings depend on hardware; `--cpu-threads` is configurable.
CUDA timings synchronize the device and include compilation and graph capture
within the training call.

A local macOS arm64 run with Python 3.12.12 and PyTorch 2.14.1 reached **93.6%**
held-out accuracy in **1.34 s** for the native training call with these defaults.
Use the saved environment and timings from your own run for comparisons.
