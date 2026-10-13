# snnsim tests

Tests are kept in one flat directory. Module names identify the subsystem or
behaviour under test; pytest markers answer **how the test runs**.

Keep a test module focused on one public API or coherent behaviour, and give new
modules specific names that remain unambiguous in this directory. Split a module
when independent behaviours accumulate, normally before it grows beyond roughly
500 lines. Several closely related tests in one module are expected. Use test
classes only when the class gives those tests a meaningful behavioural name or
shared setup.

The registered cross-cutting markers are:

- `integration`: crosses module or process boundaries.
- `brian2`: compares against the independent Brian2 simulator.
- `snntorch`: compares current-based dynamics against independent snnTorch neurons.
- `accelerator`: requires or validates MPS or CUDA behaviour.
- `slow`: belongs outside the fast iteration lane.
- `regression`: checks against calibrated artifacts.

The normal fast lane is:

```sh
uv run pytest tests/sim -m "not slow"
```

Markers compose, so the Brian2 comparisons can be selected with
`-m brian2`, while all currently marked cross-module tests can be selected with
`-m integration`.

## snnTorch comparisons

`test_snntorch_conformance.py` checks current-based LIF voltage and spike
trajectories against `snntorch.Leaky`, and compiled exponential-current networks
against `snntorch.Synaptic`, including excitation, inhibition, batch state and
split-run continuation. It also compares subthreshold input gradients and the
slope-one fast-sigmoid spike derivative.

The reference uses membrane voltage relative to rest, physically scaled current,
immediate zero reset, and zero refractory duration. These checks do not establish
conductance-based, refractory, reset-gradient or full training equivalence.
Threshold equality differs between libraries and is excluded from the matched
trajectory protocols.

Install the development dependencies with `uv sync --dev`, then run:

```sh
uv run pytest tests/sim -m snntorch
```

snnTorch is a development dependency; the comparison module skips when it is
absent, just like the Brian2 module.
