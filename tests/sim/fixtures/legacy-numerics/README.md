# Preserved numerical reference

These immutable CPU arrays were captured from SNNLab commit
`dbe0f44071601f102312b902ca8d6e75424dd60e` before COBANet removal, using that
commit's `tests/sim/test_numeric_conformance.py`. Every original graph-versus-
COBANet comparison passed before capture. `manifest.json` records PyTorch's
version, source identity and SHA-256 for each NPZ; tests verify those hashes.

The ten forward fixtures cover active and isolated recurrence at 0.05, 0.1,
0.2, 0.3 and 0.6 ms. They retain parameters, E/I spikes and voltages, three
conductance trajectories and mean pre-reset-voltage logits. The backward
fixture retains four cross-entropy losses, final gradients, constrained
parameters and AdamW state. Keys use the current semantic graph parameter IDs.

Forward spikes, conductances and parameters require exact equality. Voltages
and logits use absolute and relative tolerances of `1e-6`: float32 voltage
rounding can differ between the ARM CPU that captured the fixtures and x86
CPUs. The backward fixture uses the same numeric tolerances; continuation on
the current platform still requires exact equality. Shapes and dtypes always
match exactly. Failed forward reports include measured absolute/relative errors.

To reconstruct the reference, inspect the pinned source with Git and capture
the reference argument passed to `compare_conformance_layers` by the two
legacy/graph tests. Do not regenerate expected arrays from the current graph
implementation merely to accommodate a failure. New behavior requires an
explicitly reviewed change to the numerical contract.
