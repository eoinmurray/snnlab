# Extension validation benchmark

Run from a development checkout:

```sh
uv run python benchmarks/extension_validation.py --device cuda --warmups 3 --repetitions 15
```

The benchmark builds a synthetic 784 → 256 → 10 custom current-neuron graph with 50 timesteps and batch size 128. It compares strict and structural validation with identical weights and preencoded random input. It verifies class-score and parameter-gradient agreement (`atol=1e-5`, `rtol=1e-4`) before timing. The custom neuron owns a current filter; projections also retain their nonzero exponential decay. This is a representative validation workload, not a reproduction of the MNIST/snnTorch experiment in issue 13.

1. Four CPU threads, float32 and disabled CUDA TF32 are fixed. JSON output records PyTorch version, device, GPU name, warmups and repetitions.
2. Each timed operation includes forward initialization and either inference or forward/cross-entropy/backward. Inputs, construction, gradient zeroing, optimizer steps and profiling are excluded. Condition/operation order is randomized; device synchronization brackets timing.
3. Operator profiles are collected separately after timing. Counts include forward initialization and the explicit final synchronization. Structural validation therefore still has finite scans and scalar reads at initialization; it removes the checks that grow with timestep count.
4. Median timings are diagnostic observations, not full-training speedups. Measure CUDA profiles and unprofiled timings on the target GPU, then rerun the independent training/test protocol before making end-to-end performance claims. User callbacks and other executor paths can introduce their own synchronization.

A local CPU smoke run on PyTorch 2.14.1 (one warmup, three repetitions) found strict forward/forward-backward scalar-read counts of 510/610 and structural counts of 10/10, with score/gradient agreement. It establishes the host-read mechanism on this synthetic workload; no CUDA timing result was obtained locally.
