# General examples

Small self-contained CPU examples for the snnlab workflow. Run from the repository root after `uv sync`:

1. `uv run python examples/build_bundle.py` authors, writes and reloads a circuit bundle.
2. `uv run python examples/poisson_circuit.py` drives an E/I circuit with Poisson input.
3. `uv run python examples/replay_inputs.py` compares dense and sparse representations of the same spikes.
4. `uv run python examples/train_readout.py` trains and evaluates a small synthetic readout fixture.
5. `uv run python examples/resume_training.py` checks uninterrupted versus resumed CPU training.
6. `uv run python examples/plot_recording.py` plots retained spikes and membrane voltage.

Outputs are written under `artifacts/examples/` and replaced on repeated runs. No datasets, Graphviz or FFmpeg are required. The examples are interface demonstrations, not scientific validation or held-out accuracy results.

Run all examples with assertions in a temporary directory:

```sh
uv run python docs/scripts/check_examples.py
```

The documentation pages and downloads are generated from these files:

```sh
python3 docs/scripts/generate_examples.py
python3 docs/scripts/generate_examples.py --check
```

Edit explanatory notes in `docs/scripts/example_notes.json`. The published examples start at https://ssnlab.eoinmurray.info/examples/.
