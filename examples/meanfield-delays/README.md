# Coupled E/I circuits with explicit transmission delays

Run `uv run python examples/meanfield-delays/example.py` from the repository root.
Use `--out-dir /absolute/path` to select another output directory. Output files
are replaced on each run.

The four populations E_A, I_A, E_B and I_B have 0.1 ms internal delays and 2 ms
reciprocal E_A → E_B / E_B → E_A delays. AMPA decay is 3 ms and GABA decay is
5 ms: transmission delay and synaptic decay are distinct quantities. A graph
`StepGain` enables cross-circuit coupling at 500 ms; reduction explicitly selects
`delays="explicit", modulation="translate"`. Gains multiply activity at target
delivery, as in the graph executor. Rates and conductances remain continuous.

Each population has constant applied current; E_B receives slightly greater
current than E_A. These illustrative values are not fitted to a spiking network
and do not establish uncoupled frequency detuning or sustained gamma.

1. `network.png` shows the authored graph.
2. `trajectory.png` compares the switched and uncoupled rate trajectories.
3. `model.json` records equations, graph delays, gain controls and assumptions.
4. `switched.json` and `uncoupled.json` record named trajectories, constant history,
   parameters, solver tolerances, interpolation and integration settings.
5. `analysis.json` records the onset, delays and largest rate change between runs.

Both runs use the same non-equilibrium initial state and constant history. The
uncoupled comparison overrides both cross-circuit controls with zero throughout.
The comparison is descriptive; no locking or bifurcation inference is performed.
SNNLab owns the reusable integration; exp133 provenance, phase extraction, sweeps
and locking maps remain downstream work. Removing the delay approximation does
not validate the stationary gain/noise closure.
