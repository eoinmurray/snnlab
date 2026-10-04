# snnlab

A Python library for authoring, simulating and visualising conductance-based spiking neural networks.

```sh
uv add git+https://github.com/eoinmurray/snnlab
```

```python
from snnlab import lang, sim, viz
```

1. `snnlab.lang` authors and validates deterministic graph bundles.
2. `snnlab.sim` executes graphs and supports surrogate-gradient training.
3. `snnlab.viz` renders recordings, diagrams, figures and animations.

The simulator currently supports COBA-LIF and leaky-integrator graph populations, AMPA/GABA projections, recurrent/feedback connections and integer-timestep delays. It is not an arbitrary-equation simulator.

## Simulator commands

```sh
uv run snnsim --help
uv run python -m snnlab.sim sim --help
uv run python -m snnlab.lang.examples.build_examples
```

Graphviz (`dot`) is required for diagram exports. FFmpeg is required for video exports. Install these separately through your operating system package manager. Core authoring and simulation do not require either executable.

## Development

```sh
uv sync --dev
uv run pytest -m "not slow"
```

For development in Pinglab, use `uv add --editable ../snnlab`. For reproducible runs, use a Git tag or commit and commit the consumer lockfile.

## Documentation

The Fumadocs site lives in [`docs/`](docs/README.md), with guides for authoring, simulation, training, visualisation, and scientific contracts.

```sh
cd docs
npm ci
npm run dev
```

## Compatibility

Release history is recorded in [CHANGELOG.md](CHANGELOG.md). See
[VERSIONING.md](VERSIONING.md) for the version policy, release preparation and
tagging workflow. Check release metadata with `uv run python scripts/version.py check`.

This initial extraction retains the existing bundle schemas, backend target `tools/snnsim`, component format versions and numerical defaults. Those strings identify persisted scientific contracts; they are not Python import paths. Package version 0.1.0 identifies the combined distribution.

The retained-data regression against historical Pinglab runs remains in Pinglab. The portable tests live here. No scientific runs or generated example bundles are shipped.

MIT licensed.
