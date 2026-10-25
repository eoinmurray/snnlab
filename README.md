# snnlab

A Python library for authoring, simulating and visualising spiking neural networks with conductance-based and current-based dynamics.

```sh
uv add snnlab
```

```python
from snnlab import analysis, lang, meanfield, sim, viz
```

1. `snnlab.lang` authors and validates deterministic graph bundles.
2. `snnlab.sim` executes graphs and supports surrogate-gradient training.
3. `snnlab.viz` renders recordings, diagrams, figures and animations.
4. `snnlab.analysis` measures population activity, rhythmicity and conductance trajectories from recorded NumPy data.
5. `snnlab.extensions` registers versioned Python definitions for custom dynamics, weights, operations, training and encoders. Saved bundles retain import references and automatically load available implementations in fresh processes, without embedding Python code.
6. `snnlab.meanfield` generates symbolic qualitative population models, with manipulation and analysis in its main namespace and approximation definitions in `meanfield.closures`. See the [module guide](src/snnlab/meanfield/README.md) and [mean-field example](examples/meanfield/README.md), plus [explicit delayed E/I coupling](examples/meanfield-delays/README.md).

The graph simulator supports LIF, ALIF and AdEx populations with current or conductance synapses, plus leaky-integrator readouts, recurrent/feedback connections and integer-timestep delays. `lang.ALIF()` and `lang.ADEX()` select current models; `lang.COBA_ALIF()` and `lang.COBA_ADEX()` select conductance models. See `examples/neurons/example.py` for a LIF/ALIF/AdEx comparison and the [neuron and synapse reference](docs/content/docs/api/extensions.mdx) and [weight reference](docs/content/docs/api/lang.mdx). Register importable Python callbacks when authoring custom models; bundle loaders automatically resolve their saved references in each execution process. See `examples/current-lif/example.py` for a built-in current-based simulation and `examples/customisation/example.py` with its callbacks in the same file for a custom adaptive current neuron and weight distribution.

For a single-compartment Hodgkin–Huxley neuron with sodium, potassium and leak conductances, run `uv run python examples/hodgkin-huxley/example.py`. The [extension example](examples/hodgkin-huxley/example.py) evolves voltage and channel gates continuously without a spike reset, and plots its current-pulse response. The [Hodgkin–Huxley example guide](docs/content/docs/hodgkin-huxley.mdx) gives the equations and a reproducible independent SciPy comparison. It demonstrates forward simulation rather than surrogate-gradient training.

New same-population projections exclude self-connections by default. Use `Network.connect(..., allow_autapses=True)` for deliberate self-feedback. Exclusion persists through training and checkpoints; historical bundles without the policy retain their original behaviour. See [Weights](docs/content/docs/api/lang.mdx) for eligible fan-in, initialization and compatibility.

The simulator executes authored graph bundles. Legacy Config/COBANet execution,
flag-built networks and config replay have been removed; see the
[CLI reference](docs/content/docs/api/sim.mdx) for current inputs and controls.

Each directory under `examples/` has one `example.py`. Run it with `uv run python examples/<name>/example.py`; generated files go into its `results/` directory, which is replaced on every run. Inference and PyTorch reuse the saved model in `examples/training/results/`.

The [diagram gallery](examples/diagrams/README.md) renders nine network cases in circuit and training views, with SVG and PNG downloads. Run `uv run python examples/diagrams/example.py`, then open `examples/diagrams/results/index.html`. Graphviz’s `dot` and `neato` executables must be on `PATH`.

Native training also supports online minibatch metrics, validation-only epoch evaluation, stratified schedules with configurable tails, and immutable epoch progress callbacks. See `examples/native-training/example.py` for partial checkpoint resume and final/selected checkpoints, and the [execution policies](docs/content/docs/api/sim#inputs-and-execution-settings.mdx) for measurement and RNG contracts.

Use `sim.TrainingHistory` and `sim.NetworkSnapshot` to read native experiment evidence,
`sim.DatasetSnapshot` for named partitions with original sample identities, and explicit
result exporters for portable metrics/recordings. NumPy results include retained recordings
with per-signal coordinates; `viz.plot_raster` and `viz.plot_learning_curves` draw on your axes.
See the [experiment utilities](docs/content/docs/api/sim#traininghistory.mdx) for
configuration, readers and exporters.

For a quick offline training benchmark with conductance-based LIF neurons, run
`uv run python benchmarks/scikit-digits/benchmark.py`. See the
[scikit-learn digits benchmark](benchmarks/scikit-digits/README.md) for the
protocol, smoke mode and saved metrics/checkpoints.

Spatial inputs and populations can declare `spatial_shape=(channels, height, width)` while retaining flat execution tensors. `lang.Conv2d` supplies shared-kernel connectivity for current or conductance projections, and `lang.pool2d` supplies average/max spatial pooling. See the [spatial API](docs/content/docs/api/lang#constructor.mdx#convolutional-connectivity) for geometry, compact kernel layout and training support.

## Simulator commands

```sh
uv run snnsim --help
uv run python -m snnlab.sim sim --help
```

Graphviz (`dot`) is required for diagram exports. FFmpeg is required for video exports. Install these separately through your operating system package manager. Core authoring and simulation do not require either executable.

## Development

```sh
uv sync --dev
uv run pytest -m "not slow"
```

The full Python test suite runs nightly on `main` at 03:17 Europe/London time.
Pushes and pull requests do not trigger package tests; release checks run lint
and build the package without running the tests.

For failure emails, enable Email and Only notify for failed workflows under
System → Actions in [GitHub notification settings](https://github.com/settings/notifications).
GitHub sends scheduled workflow notifications to the account that created or
last changed the schedule. Public-repository schedules are disabled after
60 days without repository activity.

For development in Pinglab, use `uv add --editable ../snnlab`. For reproducible runs, use a Git tag or commit and commit the consumer lockfile.

## Documentation

The Fumadocs site lives in [`docs/`](docs/README.md), with Installation and Quickstart guides.

```sh
cd docs
bun install --frozen-lockfile
bun run dev
```

## Compatibility

Release history is recorded in [CHANGELOG.md](CHANGELOG.md). The package version
is defined in `src/snnlab/__init__.py`; Hatchling reads it when building.

To prepare a release, update that version and move the Unreleased changelog notes
into a dated section matching the new version. Use patch versions for compatible
fixes and minor versions for new functionality or breaking changes before 1.0.
Changes to public APIs, numerical defaults or persisted formats need explicit
changelog notes.

After checks pass, pushing the version change to `main` triggers the publishing
workflow, which publishes to PyPI and creates the matching `v<VERSION>` Git tag.
Publishing requires the repository's `pypi` environment and a PyPI Trusted Publisher
configured for `.github/workflows/publish.yml`. The workflow can also be run
manually to retry a release.

For an explicitly requested release without running documentation tests, use a
`[skip ci]` commit to suppress automatic push workflows, then dispatch
`publish.yml` on `main` and `docs.yml` on `main` with `skip_tests=true`.
Documentation tests run by default; Python package tests run only nightly.
Manual no-test releases still run lint, package builds and documentation
compilation.

The library retains the existing bundle schemas, backend target `tools/snnsim`, component format versions and numerical defaults. Those strings identify persisted scientific contracts; they are not Python import paths. The combined distribution version is defined in `src/snnlab/__init__.py`; component format versions identify separate compatibility contracts.

The retained-data regression against historical Pinglab runs remains in Pinglab. The portable tests live here. No scientific runs or generated example bundles are shipped.

MIT licensed.
