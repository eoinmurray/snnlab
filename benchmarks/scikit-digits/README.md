# Scikit-learn digits / conductance LIF

A small CPU training benchmark using the 1,797 bundled 8×8 handwritten digits
from `sklearn.datasets.load_digits`. No downloads, GPU, compilation warmup, or
extra dependencies are needed.

Run from the repository root after `uv sync --dev`:

```sh
uv run python benchmarks/scikit-digits/benchmark.py
uv run python benchmarks/scikit-digits/benchmark.py --smoke
uv run python benchmarks/scikit-digits/benchmark.py --epochs 30 --output /tmp/digits-run
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

Each run creates a new directory under `artifacts/` (or `--output`, which must
not exist), containing:

- `results.json`: configuration, dataset hash and split indices, encoding seeds,
  software/platform information, epoch losses/accuracies, native phase timings,
  update count and overall timing.
- `network.bundle/`: portable built-in graph and training recipe.
- `final.checkpoint/`: native final training checkpoint for inference or resume.

`native_train_s` covers the native trainer, including setup, initial and epoch
evaluations, and checkpoint writing. `workflow_s` additionally covers loading,
encoding and graph/bundle creation. Both exclude Python imports and the final
JSON write. Timings depend on hardware; `--cpu-threads` is configurable.

A local macOS arm64 run with Python 3.12.12 and PyTorch 2.14.1 reached **93.6%**
held-out accuracy in **1.34 s** for the native training call with these defaults.
Use the saved environment and timings from your own run for comparisons.
