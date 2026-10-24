"""Private persistent state for prepared sessions using existing network blocks."""

from __future__ import annotations

import torch

from snnlab._shared.gains import sample as sample_gain
from snnlab.sim.graph.precision import execution_precision
from snnlab.sim.runtime.state import GraphRuntimeState


class PreparedBlocks:
    """Run functional numerical blocks; snapshot only at an explicit boundary.

    Kernel inputs are never mutated, so initial state can be reused on reset and
    a failed invocation cannot partially commit a new trajectory. Output copies
    remain the session's responsibility.
    """

    def __init__(self, model, context, *, steps, dtype, device):
        self.model = model
        self.kernel = context["kernel"]
        self.initial = context["state"]
        self.state = self.initial
        self.completed_steps = 0
        self.steps, self.dtype, self.device = steps, dtype, device
        self.gain_specs = context["gain_specs"]
        self.ports = context["ports"]
        self.weights = model._stream_metadata["effective_weights"]
        self.outputs = tuple((r["id"], r["signal"]) for r in model.plan.outputs)
        self.parameter = next(model.parameters(), None)
        input_signals = {
            f"{row['id']}.value": row["id"]
            for row in model.plan.graph.get("inputs", ())
        }
        self.output_sources = tuple(
            (name, input_signals.get(signal), signal) for name, signal in self.outputs
        )

    def reset(self, context=None):
        self.state = self.initial if context is None else context["state"]
        self.completed_steps = 0 if context is None else context["completed_steps"]

    def execute(self, inputs, probe):
        voltage, refractory, adaptive, conductance, histories, input_histories = (
            self.state
        )
        gains = {
            name: torch.as_tensor(
                sample_gain(
                    spec,
                    start_step=self.completed_steps,
                    steps=self.steps,
                    dt_ms=self.model.plan.dt_ms,
                ),
                device=self.device,
                dtype=self.dtype,
            )
            for name, spec in self.gain_specs.items()
        }
        # The common window fits one numerical block. Pass the existing input
        # tensors directly, avoiding views and temporary chunk/list dictionaries.
        if self.steps <= self.model.compile_block_steps:
            if probe is not None:
                probe.mark("Compiled numerical block")
            with execution_precision(self.model.precision, self.parameter):
                result = self.kernel(inputs, *self.state, self.weights, gains, {})
            if probe is not None:
                probe.mark("Output / history assembly")
            samples = result[-1]
            outputs = {
                name: inputs[source] if source is not None else samples[signal]
                for name, source, signal in self.output_sources
            }
            return outputs, result[:6]
        chunks = {}
        # Integrator sums belong to a window, not the continuing trajectory.
        integrator_sum = {}
        with execution_precision(self.model.precision, self.parameter):
            for begin in range(0, self.steps, self.model.compile_block_steps):
                end = min(begin + self.model.compile_block_steps, self.steps)
                if probe is not None:
                    probe.mark("Compiled numerical block")
                (
                    voltage,
                    refractory,
                    adaptive,
                    conductance,
                    histories,
                    input_histories,
                    _,
                    integrator_sum,
                    samples,
                ) = self.kernel(
                    {name: value[begin:end] for name, value in inputs.items()},
                    voltage,
                    refractory,
                    adaptive,
                    conductance,
                    histories,
                    input_histories,
                    self.weights,
                    {name: value[begin:end] for name, value in gains.items()},
                    integrator_sum,
                )
                if probe is not None:
                    probe.mark("Output / history assembly")
                for name, value in samples.items():
                    chunks.setdefault(name, []).append(value)
        signals = {
            name: values[0] if len(values) == 1 else torch.cat(values)
            for name, values in chunks.items()
        }
        signals.update({f"{name}.value": value for name, value in inputs.items()})
        outputs = {name: signals[signal] for name, signal in self.outputs}
        state = (voltage, refractory, adaptive, conductance, histories, input_histories)
        return outputs, state

    def snapshot(self):
        voltage, refractory, adaptive, conductance, histories, inputs = self.state
        metadata = self.model._stream_metadata
        return GraphRuntimeState(
            signature=metadata["signature"],
            compatibility=metadata["compatibility"],
            completed_steps=self.completed_steps,
            voltages=voltage,
            refractory=refractory,
            conductances={
                p.id: conductance[(p.id, p.polarity)]
                for p in self.model.plan.projections
                if self.ports[p.id] == "conductance"
            },
            currents={
                p.id: conductance[(p.id, p.polarity)]
                for p in self.model.plan.projections
                if self.ports[p.id] == "current"
            },
            population_histories={
                name: torch.stack(values) for name, values in histories.items()
            },
            input_histories=inputs,
            custom_state={
                f"neuron/{name}/{key}": value
                for name, state in adaptive.items()
                for key, value in state.items()
            },
        )
