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
- `accelerator`: requires or validates MPS or CUDA behaviour.
- `slow`: belongs outside the fast iteration lane.
- `regression`: checks against calibrated artifacts.

The normal fast lane is:

```sh
uv run pytest tools/snnsim/tests -m "not slow"
```

Markers compose, so the Brian2 comparisons can be selected with
`-m brian2`, while all currently marked cross-module tests can be selected with
`-m integration`.
