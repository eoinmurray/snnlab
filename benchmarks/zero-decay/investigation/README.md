# Training accuracy discrepancy investigation

This investigation follows release 0.12.0 (`69bb7013d1f4ed6826014283dcdf68e690166e54`). It executes diagnostic training workloads, without rerunning the test suite or changing released library code.

The original two-epoch MNIST benchmark reported 83.92% accuracy for snnTorch and 77.67% for all three SNNLab variants. Its equations and fixed-weight scores matched, but its backward computational operations did not match exactly.

## Identified numerical differences

1. The registered SNNLab surrogate multiplies the incoming gradient by a computed reciprocal of the denominator. snnTorch FastSigmoid divides the incoming gradient by that denominator directly. These real-number expressions round differently in float32.
2. The snnTorch benchmark broadcasts its bias tensor directly. SNNLab represents the bias as a projection from an all-ones input, so its bias gradient is calculated with a matrix multiplication rather than broadcast backward's sum. These CUDA reductions round differently.

The first original CUDA update differs by up to 1.30385160446167e-7 in a parameter, despite exactly equal class scores and a maximum gradient difference of only 9.5367431640625e-7. Scores first differ at zero-based update 11 (the twelfth minibatch). `original.json` records 160 matched updates.

`deterministic.json` repeats 40 updates with deterministic algorithms and a CUBLAS workspace configuration. Two independent snnTorch controls remain exactly equal. Replacing only the SNNLab surrogate backward eliminates initial weight-matrix gradient differences, but leaves bias differences and later trajectory divergence. Strict and fast execution have identical discrepancies in this control.

`matched.json` uses the snnTorch backward function for the diagnostic SNNLab executor and uses an explicit all-ones bias matrix product in the diagnostic snnTorch reference. This is an experimental intervention, not a library patch. The first 40 matched updates have exactly equal class scores, every parameter gradient and every updated parameter. Complete matched training results are recorded in that file.

## Full matched training result

The controlled pair processed all 60,000 MNIST training examples for two epochs and all 10,000 test examples. Both returned exactly the same reported epoch losses `[1.279261403242747, 0.5599448835372925]`, test loss `0.4711558293104172`, and **87.20% test accuracy**.

This confirms that matching these backward computational operations removes the discrepancy in this controlled configuration. It supports a numerical explanation for the original trajectory divergence. The intervention also enables deterministic CUDA algorithms and `CUBLAS_WORKSPACE_CONFIG=:4096:8`; its absolute accuracy should not be attributed to the bias operation alone. This single-seed result does not establish general accuracy equivalence or superiority. The strict/fast zero-decay speedup did not cause the original gap: their original losses/accuracy and diagnostic updates match.

Modal preempted the combined job after the reference completed. Its result was retained in `reference-run.txt`; the diagnostic trace and SNNLab full run were completed in separate calls. The final JSON links all runs. Timing these control runs is outside the original performance protocol and should not replace its timing results.

No released library code was changed. These new investigation files are retained separately for review.

## Why the comparisons missed it

1. The committed snnTorch conformance suite checks neuron dynamics and individual derivatives, rather than sustained classifier optimization. Its unrolled gradient case is float64, subthreshold and excludes reset differentiation. Its spike derivative comparison uses slope one and sums spikes, giving an incoming gradient of one. It does not exercise this slope-25, float32, spiking classifier with clipped Adam updates.
2. The zero-decay benchmark checks exact scores and tolerant gradients at two shared checkpoints, including a checkpoint trained by the reference. Its gradient tolerance is `atol=1e-5, rtol=1e-4`, which accepts the initial discrepancy. It does not assert equality of independently trained trajectories or accuracies. Checking one common trained checkpoint cannot detect how separately trained weights drift.
3. Additional comparison work present in the workspace is outside the 0.12.0 release. It includes a different protocol with short synthetic training coverage; it does not establish full CUDA MNIST accuracy agreement for this benchmark.

These were insufficient acceptance checks for a claim of full training equivalence. The benchmark report explicitly withheld that claim, but finding the discrepancy should trigger this investigation rather than relying on fixed-checkpoint agreement.

## Coverage to add

1. A CUDA float32 spiking classifier regression with matched operations, nonuniform upstream gradients, clipping and at least 40 matched Adam updates, asserting gradients, optimizer state, weights, scores and losses.
2. An end-to-end matched training control with explicit accuracy/loss acceptance criteria. Keep this separate from performance timing and retain differences as failures or clearly reported limitations.
3. Multiple seeds when comparing accuracy between deliberately different arithmetic implementations. One seed's accuracy gap cannot establish an inherent quality difference between libraries.

## Reproduction

Run `modal run benchmarks/zero-decay/investigation/modal_investigation.py` with Modal authentication and the checked-out 0.12.0 source. The script uploads only the library source and the zero-decay benchmark. It uses one L4, PyTorch 2.10.0, snnTorch 0.9.4, the original seed-42 MNIST initialization and minibatch/encoding protocol, and writes `matched.json` here after each completed call to preserve completed work across preemption. This is diagnostic execution, not a test-suite invocation.
