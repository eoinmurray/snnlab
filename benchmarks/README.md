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

# GPU execution efficiency

The paired benchmark compares baseline commit `55de871d597d9ca915c27c747d642e93deec9ed0`
(0.10.1) with the 0.11.0 implementation. It uses a single NVIDIA L4 on Modal,
PyTorch 2.10.0+cu128 (the package's minimum supported version), four CPU threads,
float32 and disabled TF32. See `gpu-efficiency-results.json` for complete timing
samples, cold calls, peak/resident/incremental allocated memory and operator counts.

1. The primary graph is 784 → 256 → 10, with CUBA LIF, exponential currents,
   32 timesteps and batch size 128. Additional cases use hidden/batch sizes 64/8
   and 256/32. Baseline and candidate score, gradient, training-parameter and
   loss payloads agree (`atol=2e-5`, `rtol=2e-5`). Prefetched inputs/results agree
   exactly with the synchronous candidate. The acceptance tests also cover all
   six spiking models, delayed recurrence, autapses and continuation.
2. Each operation has a separately reported cold call, two warmups and seven
   synchronised timing samples. Profiles are collected outside timing. Conditions
   run sequentially; these are diagnostic measurements on one shared cloud GPU,
   not randomized estimates of general speedup. Construction/checkpoint copies
   are included in the complete training request; zeroing is included in the
   forward/backward operation. Dataset timings include snapshot resolution,
   validation, encoding, transfers and execution.
3. `cuda_activity_count` includes GPU kernels and other CUDA profiler activities,
   including copies; it is not exclusively a compute-kernel count.
   The host profiler does not capture every worker-thread preparation operation;
   evaluate prefetched performance using the unprofiled total times and tests.
4. In the primary candidate, compiled inference was 26.88 ms versus 30.76 ms eager,
   and compiled forward/backward was 31.62 ms versus 59.68 ms eager. The compiled
   cold calls were 5.65 s and 2.59 s respectively. Smaller-shape results are included
   in JSON. These timings justify opt-in compilation for repeated workloads,
   not a claim that every short training/inference request benefits.
5. Dataset prefetch depth 2 took 1133.76 ms versus 1344.13 ms at depth 0.
   Incremental peak allocated GPU memory rose from about 29.3 MB to 41.9 MB.
   Recording blocks of 32 took 57.96 ms versus 80.44 ms for one-step blocks,
   with the recording workload's host stream-synchronisation count falling from
   129 to 67. Event extraction calls fell from 32 to 1. Larger blocks retain more
   device samples; existing sinks keep the default block size of 1.
6. The complete eager eight-update training request measured 491.72 ms versus
   516.18 ms baseline in the final run. An earlier run was slower than baseline,
   so no consistent full-training speedup is established. The mechanism is verified:
   clones fell from 178 to 86, scalar reads from 62 to 54 and host stream
   synchronisations from 36 to 29. Selection still reads batch loss. Observation
   callbacks and strict/custom value validation can introduce additional reads.
7. The timing check reported 120.41 ms for synchronised input resolution/execution
   versus 125.34 ms externally including build. The scopes intentionally differ.
   Python allocation tracking is distinct from the CUDA allocator measurements.

Run a candidate benchmark on an existing CUDA machine:

```sh
uv run python benchmarks/gpu_efficiency.py --output gpu-results.json
```

For a paired Modal run, install the optional Modal CLI in a separate environment
and authenticate it. Prepare a baseline directory containing its `src/` tree,
then run:

```sh
SNNLAB_BASELINE_DIR=/path/to/baseline modal run benchmarks/modal_gpu.py
SNNLAB_BASELINE_DIR=/path/to/baseline modal run benchmarks/modal_gpu.py --only-tests --output cuda-tests.json
```

The runner explicitly uploads `src`, `tests`, `benchmarks` and the baseline `src`
tree. It never uploads account configuration, other workspace directories or
external datasets. It uses one L4 per run and a 20-minute timeout. The saved run
was [Modal CUDA validation](https://modal.com/apps/me-93863/main/ap-yfbt1j5W3k0DyHI3dYRpQ5).

Final release checks: 818 CPU tests passed with 11 accelerator skips on PyTorch
2.14.1. The final [CUDA acceptance run](https://modal.com/apps/me-93863/main/ap-lru7zZW6rtO2HBIO0G5dcC)
passed 135 tests on PyTorch 2.10.0+cu128, including optimizer updates and explicit
rejection of CUDA-graph compilation modes. The latter failed an experimental
recurrent-state check and is not part of the released API. Default compilation
is the benchmarked and supported mode. Final test output is retained in the JSON.
