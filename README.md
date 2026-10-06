# snnlab

A Python library for authoring, simulating and visualising spiking neural networks with conductance-based and current-based dynamics.

```sh
uv add snnlab
```

```python
from snnlab import analysis, lang, sim, viz
```

1. `snnlab.lang` authors and validates deterministic graph bundles.
2. `snnlab.sim` executes graphs and supports surrogate-gradient training.
3. `snnlab.viz` renders recordings, diagrams, figures and animations.
4. `snnlab.analysis` measures population activity, rhythmicity and conductance trajectories from recorded NumPy data.
5. `snnlab.extensions` registers versioned Python definitions for custom dynamics, weights, operations, training and encoders. Saved bundles retain import references and automatically load available implementations in fresh processes, without embedding Python code.

The graph simulator supports COBA-LIF, current-based LIF and leaky-integrator populations, conductance and exponential-current synapses, recurrent/feedback connections and integer-timestep delays. Register importable Python callbacks when authoring custom models; bundle loaders automatically resolve their saved references in each execution process. See `examples/current-lif/current_lif.py` for a built-in current-based simulation and `examples/customisation/customisation.py` with its sibling `custom_neurons.py` for an adaptive current neuron and custom weight distribution. Run the Customisation script with `--simulate` to execute its saved bundle without rebuilding or manually registering callbacks.

The simulator executes authored graph bundles. Legacy Config/COBANet execution,
flag-built networks and config replay have been removed; see the
[CLI reference](docs/content/docs/api/sim/cli.mdx) for current inputs and controls.

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

For an explicitly requested release without rerunning tests, use a `[skip ci]`
commit to suppress automatic push workflows, then dispatch `publish.yml` and
`docs.yml` on `main` with `skip_tests=true`. Their default remains `false`, so
ordinary releases and automatic workflows continue to run tests. Manual
no-test releases still run lint, package builds and documentation compilation.

This initial extraction retains the existing bundle schemas, backend target `tools/snnsim`, component format versions and numerical defaults. Those strings identify persisted scientific contracts; they are not Python import paths. Package version 0.1.0 identifies the combined distribution.

The retained-data regression against historical Pinglab runs remains in Pinglab. The portable tests live here. No scientific runs or generated example bundles are shipped.

MIT licensed.
