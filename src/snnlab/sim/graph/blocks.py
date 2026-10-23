"""Tensor-only bounded network blocks using the reference numerical kernels.

This is the opt-in projection/neuron fusion path tracked by issue #22. Host
recording, interventions, strict callbacks and online inference stay eager.
"""

from __future__ import annotations

import torch

from snnlab._shared import neurons as N
from snnlab._shared.spatial import convolution_drive, pool_tensor, signal_spatial_shape
from snnlab.sim.graph.kernels import neuron_kernel, prepare_kernel, synapse_kernel


def signal_parameters(plan):
    """Conservative parameter reachability, including recurrent dependencies."""
    dependencies = {}
    populations = {pop["id"]: pop for pop in plan.populations}
    for pop in plan.populations:
        for port in ("voltage", "spikes", "pre_reset_voltage", "adaptation"):
            dependencies[f"{pop['id']}.{port}"] = set()
    for projection in plan.projections:
        dependencies[projection.source] = set()
    changed = True
    while changed:
        changed = False
        for projection in plan.projections:
            parameters = (
                ({projection.parameter} | dependencies.get(projection.source, set()))
                if projection.enabled
                else set()
            )
            for port in ("current", "conductance"):
                key = f"{projection.id}.{port}"
                before = dependencies.setdefault(key, set()).copy()
                dependencies[key].update(parameters)
                changed |= before != dependencies[key]
            target = projection.target.partition(".")[0]
            for port in ("voltage", "spikes", "pre_reset_voltage", "adaptation"):
                if (
                    populations[target]["neuron"]["kind"] == "leaky_integrator"
                    and projection.polarity == "inhibitory"
                ):
                    continue
                if (
                    port == "spikes"
                    and not populations[target]["spiking"]
                    and populations[target]["neuron"]["kind"]
                    in {*N.ADAPTIVE_KINDS, "leaky_integrator"}
                ):
                    continue
                key = f"{target}.{port}"
                before = dependencies[key].copy()
                dependencies[key].update(parameters)
                changed |= before != dependencies[key]
        for operation in plan.graph.get("operations", ()):
            key = f"{operation['id']}.value"
            parameters = set(operation.get("parameters", ()))
            for source in operation["sources"]:
                parameters.update(dependencies.get(source, ()))
            before = dependencies.setdefault(key, set()).copy()
            dependencies[key].update(parameters)
            changed |= before != dependencies[key]
    return dependencies


def network_block(plan, *, spike_function, projection_mode, signals, compile_mode):
    """Resolve topology once; the returned callable owns no trajectory state."""
    populations = tuple(plan.populations)
    pop_ids = {p["id"] for p in populations}
    projection_rows = {p["id"]: p for p in plan.graph["projections"]}
    incoming = {
        p["id"]: tuple(
            edge
            for edge in plan.projections
            if edge.target.partition(".")[0] == p["id"]
        )
        for p in populations
    }
    neurons = {
        p["id"]: neuron_kernel(
            p,
            dt_ms=plan.dt_ms,
            refractory=plan.refractory.get(p["id"], {}),
            spike_function=spike_function,
            compile_mode=None,
        )
        for p in populations
    }
    synapses = {
        p.id: synapse_kernel(
            p,
            spec=projection_rows[p.id]["synapse"],
            dt_ms=plan.dt_ms,
            compile_mode=None,
            projection_mode=projection_mode,
        )
        for p in plan.projections
    }
    pools = {
        f"{row['id']}.value": row
        for row in plan.graph.get("operations", ())
        if row["kind"] == "pool2d"
    }
    sources = {}
    spatial = {}
    for projection in plan.projections:
        signal, chain = projection.source, []
        while signal in pools:
            op = pools[signal]
            chain.append(
                (signal_spatial_shape(plan.graph, op["sources"][0]), op["config"])
            )
            signal = op["sources"][0]
        sources[projection.id] = signal.partition(".")[0], tuple(reversed(chain))
        config = projection_rows[projection.id].get("connectivity")
        if config is not None:
            spatial[projection.id] = (
                signal_spatial_shape(plan.graph, projection.source),
                config,
            )
    ports = {
        p.id: "current"
        if projection_rows[p.id]["synapse"]["kind"] == "exponential_current"
        else "conductance"
        for p in plan.projections
    }

    # Only sinks may be deferred: even delayed feedback makes a population
    # part of the causal core. Resolve pooled sources before checking edges.
    outgoing = {sources[p.id][0] for p in plan.projections if p.enabled}
    terminal = tuple(p for p in populations if p["id"] not in outgoing)
    core = tuple(p for p in populations if p["id"] in outgoing)
    stages = tuple(stage for stage in (core, terminal) if stage)

    def block(
        inputs,
        voltage,
        refractory,
        adaptive,
        conductance,
        histories,
        input_histories,
        weights,
        gains,
        integrator_sum,
    ):
        # Functional containers keep state passed by the caller intact. No
        # detach between blocks: delays and adaptation retain BPTT gradients.
        voltage, refractory, adaptive = dict(voltage), dict(refractory), dict(adaptive)
        conductance = dict(conductance)
        histories = {name: list(values) for name, values in histories.items()}
        integrator_sum = dict(integrator_sum)
        samples = {name: [] for name in signals}
        final_spikes = {}
        trajectories = {}
        initial_histories = {name: list(values) for name, values in histories.items()}
        steps = next(iter(inputs.values())).shape[0]
        for stage in stages:
            stage_ids = {p["id"] for p in stage}
            stage_edges = tuple(
                p for p in plan.projections if p.target.partition(".")[0] in stage_ids
            )
            batched = {}
            for projection in stage_edges:
                owner, chain = sources[projection.id]
                if (
                    not projection.enabled
                    or chain
                    or projection.id in spatial
                    or owner in stage_ids
                ):
                    continue
                delay = projection.delay_steps
                # A previous stage has a complete spike trajectory. External
                # inputs are already available for the entire bounded window.
                source_steps = []
                for t in range(steps):
                    if owner in pop_ids:
                        source = (
                            trajectories[owner][t - delay]
                            if t >= delay
                            else initial_histories[owner][t - delay]
                        )
                    else:
                        source = (
                            inputs[owner][t - delay]
                            if t >= delay
                            else input_histories[owner][t - delay]
                        )
                    source_steps.append(source)
                source = torch.stack(source_steps)
                drive = source.flatten(0, 1) @ weights[projection.parameter]
                batched[projection.id] = drive.reshape(
                    steps, source.shape[1], drive.shape[-1]
                )
            for t in range(steps):
                spikes = {}
                values = {}
                for pop in stage:
                    name = pop["id"]
                    excitation = torch.zeros_like(voltage[name])
                    inhibition = torch.zeros_like(voltage[name])
                    for projection in incoming[name]:
                        key = (projection.id, projection.polarity)
                        if not projection.enabled:
                            conductance[key] = torch.zeros_like(conductance[key])
                            continue
                        owner, chain = sources[projection.id]
                        delay = projection.delay_steps
                        if projection.id in batched:
                            drive = batched[projection.id][t]
                        else:
                            if owner in stage_ids:
                                source = (
                                    spikes[owner]
                                    if delay == 0
                                    else histories[owner][-delay]
                                )
                            elif owner in pop_ids:
                                source = (
                                    trajectories[owner][t - delay]
                                    if t >= delay
                                    else initial_histories[owner][t - delay]
                                )
                            else:
                                source = (
                                    inputs[owner][t - delay]
                                    if t >= delay
                                    else input_histories[owner][t - delay]
                                )
                            for shape, config in chain:
                                source = pool_tensor(source, shape, config)
                            if projection.id in spatial:
                                shape, config = spatial[projection.id]
                                drive = convolution_drive(
                                    source, weights[projection.parameter], shape, config
                                )
                            else:
                                drive = source @ weights[projection.parameter]
                        if projection.id in gains:
                            drive = drive * gains[projection.id][t]
                        conductance[key] = synapses[projection.id](
                            conductance[key], drive
                        )
                        if projection.polarity == "excitatory":
                            excitation = excitation + conductance[key]
                        else:
                            inhibition = inhibition + conductance[key]
                    kind = pop["neuron"]["kind"]
                    if kind in N.ADAPTIVE_KINDS:
                        state, emitted = neurons[name](
                            adaptive[name], excitation, inhibition
                        )
                        adaptive[name] = state
                        voltage[name], refractory[name] = (
                            state["voltage"],
                            state["refractory"],
                        )
                        spikes[name] = (
                            emitted if pop["spiking"] else torch.zeros_like(emitted)
                        )
                        values[f"{name}.adaptation"] = state["adaptation"]
                    elif kind == "leaky_integrator":
                        pre, voltage[name], spikes[name] = neurons[name](
                            voltage[name], excitation
                        )
                        integrator_sum[name] = (
                            integrator_sum.get(name, torch.zeros_like(pre)) + pre
                        )
                        values[f"{name}.pre_reset_voltage"] = pre
                    else:
                        voltage[name], spikes[name], refractory[name] = neurons[name](
                            voltage[name], refractory[name], excitation, inhibition
                        )
                    values[f"{name}.voltage"] = voltage[name]
                    values[f"{name}.spikes"] = spikes[name]
                    trajectories.setdefault(name, []).append(spikes[name])
                    final_spikes[name] = spikes[name]
                for projection in stage_edges:
                    values[f"{projection.id}.{ports[projection.id]}"] = conductance[
                        (projection.id, projection.polarity)
                    ]
                for signal in signals:
                    if signal in values:
                        samples[signal].append(values[signal])
                for name in histories:
                    if name in stage_ids:
                        histories[name] = histories[name][1:] + [spikes[name]]
        next_inputs = {
            name: torch.cat((value, inputs[name]), dim=0)[-value.shape[0] :]
            for name, value in input_histories.items()
        }
        return (
            voltage,
            refractory,
            adaptive,
            conductance,
            histories,
            next_inputs,
            final_spikes,
            integrator_sum,
            {name: torch.stack(values) for name, values in samples.items()},
        )

    return prepare_kernel(block, compile_mode)
