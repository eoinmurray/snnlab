# Recurrent E/I mean-field analysis

This self-contained example builds a recurrent excitatory/inhibitory network,
reduces it to a qualitative rate–conductance model and explores what happens
as external drive to inhibition decreases.

```sh
uv run python examples/meanfield/example.py
```

The script has four functions:

1. `configuration()` defines the network, mean incoming couplings, closure
   parameters, input schedule and solver settings.
2. `build_model(cfg)` authors the network with `snnlab.lang`, constructs the
   six-state mean-field model with two rates and all four recurrent conductances.
3. `run(out_dir)` follows equilibria with `mf.continuation`, refines stability
   crossings with `mf.find_stability_crossings`, integrates the scheduled
   trajectory and saves results.
4. `plot(...)` draws equilibrium rates, local stability, the input schedule
   and population-rate trajectories.

Couplings are explicit conductance sums per target. Uniform graph weights
represent those sums, with self-connections excluded; no weights or spikes
are sampled. The closure uses stationary noisy-LIF gains, fixed resting
voltage driving forces and separate E/I rate relaxation times.

The external input rate is constant. Its pathway to inhibition ramps from
full strength to zero between 800 and 1,550 ms. External conductances are
quasistatic, so the model omits their filtering lag during the ramp.
Input covariance, voltage-dependent shunting and within-population
heterogeneity are also omitted. All four recurrent conductances are integrated
as separate dynamical states.

Outputs are written under `results/`:

1. `network.png`: the authored circuit; Graphviz's `dot` executable is required.
2. `analysis.png` and `analysis.svg`: the four-panel figure.
3. `analysis.json`: configuration, state names, equilibrium continuation and
   oscillatory crossing candidates.
4. `trajectory.npz`: time in ms and six-state trajectories, with Hz rates
   and uS conductances in the order named in the analysis JSON.
5. `model.json`: the complete six-state model document, written with
   `model.write(...)`. It contains states, parameter values,
   computational expressions, assumptions and individually named LaTeX equations.
   For example, `saved["equations"]["E.rate"]["latex"]` selects the E-rate equation.

Use `--out-dir PATH` to choose another destination. Matching output files are
overwritten; unrelated files are left alone. The example reads no external
input files and executes no spiking simulation. Its qualitative analysis does
not establish quantitative agreement with a spiking network or nonlinear
Hopf criticality.

```sh
uv run pytest tests/meanfield -q
```
