# Original training comparison across five seeds

This experiment repeats the original 0.12.0 training comparison using its original, numerically unmatched surrogate and bias operations. It does not apply the diagnostic intervention that made backward operations identical.

## Preselected protocol

1. Seeds: **17, 42, 123, 314, 2026**, chosen before these results. Seed 42 repeats the original observation. Seed varies initial weights, epoch shuffling, and Bernoulli train/test encoding. Within each seed those quantities are identical across backends.
2. Source: frozen release commit `69bb7013d1f4ed6826014283dcdf68e690166e54`, including the original `benchmarks/zero_decay.py`. This avoids concurrent workspace changes and the later agreement gate while deliberately measuring the known discrepancy. The uploaded benchmark SHA-256 and initial/final weight SHA-256 values are retained in each trial.
3. Conditions: original snnTorch, SNNLab strict/full snapshots, SNNLab fast/full snapshots, and SNNLab fast/output-only results. Architecture 784/256/10, 50 timesteps, alpha 0.8, beta 0.9, fast-sigmoid slope 25, delayed detached subtractive reset, float32, TF32 disabled, Adam learning rate 0.001, gradient norm clipped at 1, batches of 128. Each condition trains for two complete epochs on 60,000 MNIST examples and tests all 10,000 examples.
4. Infrastructure: Modal L4, four CPU threads per trial, PyTorch 2.10.0, torchvision 0.25.0, snnTorch 0.9.4, up to four concurrent trials. Deterministic algorithms and CUBLAS workspace are not newly enabled; their actual values are recorded. Concurrent durations are excluded from performance conclusions.
5. Analysis: report every seed, paired snnTorch-minus-strict accuracy differences, aggregate accuracies, paired mean/range and win counts. An exploratory Student t interval is provided with its normality assumption and small-sample limitation. Verify initialization pairing and exact SNNLab mode agreement in metrics and final weights. Keep unfavorable or failed outcomes visible; do not tune the protocol after viewing results.

## Reproduction

Create a frozen source directory using `git archive 69bb7013d1f4ed6826014283dcdf68e690166e54 src benchmarks/zero_decay.py`, then set `SNNLAB_COMPARISON_SOURCE` to that extracted directory when running `modal run benchmarks/zero-decay/multiseed/modal_multiseed.py`. The runner saves each completed trial locally and in a Modal Volume, so a preemption repeats only unfinished work. Cached trials are indexed by the uploaded benchmark hash, seed and condition; use the stated frozen release source for cache validity.

Run `.venv/bin/python benchmarks/zero-decay/multiseed/analyse.py` after all 20 trials complete. It writes `summary.json`, `accuracy.png`, and `accuracy.svg`. Raw trial metrics remain in `results.json`.

## Results

All **20 trials completed** on [Modal](https://modal.com/apps/me-93863/main/ap-CkjRtFdDZkYE9CdRzzZw6G). Seed 42 reproduced the original accuracies exactly. No matched-backward intervention was used.

| Seed | snnTorch accuracy | SNNLab accuracy | snnTorch minus SNNLab |
| --- | ---: | ---: | ---: |
| 17 | 78.48% | 79.20% | -0.72 pp |
| 42 | 83.92% | 77.67% | +6.25 pp |
| 123 | 61.81% | 74.04% | -12.23 pp |
| 314 | 79.22% | 81.94% | -2.72 pp |
| 2026 | 79.76% | 92.10% | -12.34 pp |
| Mean | 76.64% | 80.99% | −4.35 pp |

SNNLab scored higher in four of five seeds. The original SNNLab deficit does **not persist as a consistent direction of effect**. The paired snnTorch-minus-SNNLab differences range from −12.34 to +6.25 percentage points. Their mean is −4.35 points, with an exploratory 95% Student t interval of **−14.25 to +5.54 points**. That wide interval includes zero: these five seeds do not establish superiority or equivalence.

Strict, fast/full snapshots and fast/output-only SNNLab produced **exactly identical epoch losses, test losses, accuracies and final-weight hashes for every seed**. Their shared initial-weight hashes match the snnTorch starting weights in each pair. Actual settings were PyTorch 2.10.0+cu128, snnTorch 0.9.4, NVIDIA L4, deterministic algorithms disabled, and no explicit CUBLAS workspace configuration.

Numerical implementation differences remain consequential for individual training trajectories. The original single-seed accuracy gap is not evidence of a consistent quality penalty from SNNLab or its fast execution. Two epochs and five seeds give an exploratory comparison; no hyperparameter tuning, longer training or additional seed selection was performed after observing outcomes.

See [raw results](results.json), [analysis summary](summary.json), and [paired accuracy plot](accuracy.png). The standard deviation of accuracy across these seeds was 8.55 percentage points for snnTorch and 6.83 for SNNLab; these are descriptive sample estimates, not established variability differences.

## Interpretation limits

Five seeds give a preliminary view of variability for this architecture, training duration and device family. They cannot establish general equivalence or superiority across tasks or converged training. Test spike encoding also differs between seeds, though it is paired across libraries within each seed. Exact numerical reproduction and comparable distributions of training performance are distinct questions.
