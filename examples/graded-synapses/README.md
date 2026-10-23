# Reciprocal graded inhibition

Run from the repository root:

```sh
uv run python examples/graded-synapses/graded_synapses.py
```

The example registers importable passive conductance-based membrane and graded
synapse callbacks. Two separate one-cell populations inhibit each other through
voltage-driven graph edges. This demonstrates the coupling contract; it does not
implement the intrinsic Toolbox half-centre oscillator neurons.

The script checks a 200-step CPU float64 simulation against an independent NumPy
forward-Euler loop and writes `network.png` and `trajectory.csv` beside the script.
Diagram rendering requires Graphviz's `dot` executable. CSV columns contain
both filtered voltage gates (mV), conductances (uS) and membrane voltages (mV).
Each row represents the completed gate and membrane update for one 0.1 ms step.

1. Both synapses read old membrane voltage, update their filtered gates, then
   deliver new conductance to their respective membrane updates.
2. Weights are dimensionless and act before gating. Maximum conductance is
   0.04 uS inside the callback; capacitance is 0.2 nF and reversal is -80 mV.
3. `network(state_source=True)` instead samples the previous completed custom
   `activation` state. That state intentionally differs from voltage after the
   first step, so tests detect accidental substitution or timing changes.
4. `network(delay_steps=3)` adds three sample steps, holding initial source
   values before time zero. Synaptic gates and delayed source samples persist
   through `save_runtime_state` / `load_runtime_state`.
5. A `clock` input supplies time and batch axes for this autonomous circuit.
   It is not connected to a membrane. Tests also cover batched execution,
   reordered populations, incompatible units, portable bundles and gradients.
