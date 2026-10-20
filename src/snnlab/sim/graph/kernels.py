"""Prepared tensor kernels; graph orchestration and validation remain eager."""

from __future__ import annotations

import copy
import math

import torch

from snnlab import extensions as E
from snnlab._shared import neurons as N
from snnlab.sim.graph import dynamics as M
from snnlab.sim.graph import extensions as X


def precision_fallback(device, dtype):
    return (
        torch.__version__.split("+")[0].startswith("2.10.")
        and device.type == "cpu"
        and dtype in {torch.float16, torch.bfloat16}
    )


def prepare_kernel(step, compile_mode, *, fullgraph=True):
    if compile_mode is not None:
        # Each prepared closure owns its compiler guards. Sharing a code object
        # across unrelated configurations exhausts Dynamo's cache limit.
        step.__code__ = step.__code__.replace(co_name=f"tensor_step_{id(step)}")
        options = {"emulate_precision_casts": True}
        if torch.__version__.split("+")[0].startswith("2.10."):
            # 2.10 CPU vector codegen rejects precision casts; scalar codegen
            # compiles but still changes recurrent gradients. Choose the eager
            # tensor graph at compile time, without per-step argument scans.
            def backend(graph, examples):
                if any(
                    isinstance(value, torch.Tensor)
                    and precision_fallback(value.device, value.dtype)
                    for value in examples
                ):
                    return graph.forward
                return torch._inductor.compile(graph, examples, options=options)

            return torch.compile(
                step, backend=backend, fullgraph=fullgraph, dynamic=False
            )
        return torch.compile(
            # Fusion must retain eager rounding at intermediate dtype casts.
            # In bfloat16, dropping those casts changed recurrent gradients by
            # over 30%, even for compiled individual kernels.
            step,
            options=options,
            fullgraph=fullgraph,
            dynamic=False,
        )
    return step


def neuron_kernel(
    population,
    *,
    dt_ms,
    refractory,
    spike_function,
    compile_mode="default",
    definition=None,
    fullgraph=True,
):
    config = copy.deepcopy(population["neuron"])
    kind = config["kind"]
    ref_steps = refractory.get("refractory_steps", 0)
    if kind == "custom_neuron":

        def step(state, excitatory, inhibitory):
            return definition.function(
                E.NeuronContext(
                    state,
                    excitatory,
                    inhibitory,
                    dt_ms,
                    config.get("config", {}),
                    spike_function,
                )
            )
    elif kind == "cuba_lif":

        def step(voltage, ref, excitatory, inhibitory):
            return X.current_lif(
                {"voltage": voltage, "refractory": ref},
                excitatory,
                inhibitory,
                dt_ms=dt_ms,
                config=config,
                ref_steps=ref_steps,
                spike_function=spike_function,
            )
    elif kind in N.ADAPTIVE_KINDS:

        def step(state, excitatory, inhibitory):
            return X.adaptive_neuron(
                state,
                excitatory,
                inhibitory,
                dt_ms=dt_ms,
                config=config,
                ref_steps=ref_steps,
                spike_function=spike_function,
            )
    elif kind == "leaky_integrator":
        beta = math.exp(-dt_ms / float(config["tau"]["value"]))
        threshold = config.get("soft_reset_threshold")
        slope = float(config.get("surrogate_slope", M.SURROGATE_SLOPE))

        def step(voltage, excitatory):
            pre_reset = beta * voltage + (1.0 - beta) / dt_ms * excitatory
            spikes = torch.zeros_like(voltage)
            post_reset = pre_reset
            if threshold is not None:
                reset = M.fast_sigmoid_spike(pre_reset - float(threshold), slope)
                if population.get("spiking"):
                    spikes = reset
                post_reset = pre_reset - reset * float(threshold)
            return pre_reset, post_reset, spikes
    elif kind == "coba_lif":

        def step(voltage, ref, excitatory, inhibitory):
            tau = float(config["tau_mem"]["value"])
            capacitance = float(config.get("capacitance_nf", 1.0 if tau >= 15 else 0.5))
            leak = float(config.get("leak_us", capacitance / tau))
            dampen = float(config.get("voltage_grad_dampen", M.V_GRAD_DAMPEN))
            threshold = float(config.get("threshold_mv", M.V_th))
            following = refractory["timing"] == N.REFRACTORY_TIMING
            return M.lif_step_expeuler(
                voltage,
                ref,
                excitatory,
                inhibitory,
                capacitance,
                leak,
                ref_steps,
                lambda value, threshold_offset=0.0: spike_function(
                    value - threshold - threshold_offset
                ),
                dt_override=dt_ms,
                v_grad_dampen=dampen,
                refractory_after_step=following,
            )
    else:
        return None
    return prepare_kernel(step, compile_mode, fullgraph=fullgraph)


def synapse_kernel(
    projection, *, spec, dt_ms, compile_mode, projection_mode, definition=None
):
    config = copy.deepcopy(spec.get("config", {}))
    if definition is not None:

        def step(state, drive):
            return definition.function(E.SynapseContext(state, drive, dt_ms, config))
    elif projection_mode == "fast" and projection.decay == 0.0:

        def step(previous, drive):
            # Autocast matrix products may be BF16; recurrent synaptic state
            # retains its declared dtype, including the zero-decay fast path.
            return drive.to(previous.dtype)
    else:
        decay = projection.decay

        def step(previous, drive):
            return previous * decay + drive

    return prepare_kernel(step, compile_mode)


def constraint_kernel(spec, *, compile_mode, check_values=True):
    if not spec:
        return lambda value: value
    spec = copy.deepcopy(spec)
    if spec["kind"] == "non_negative":

        def step(value):
            return value.clamp(min=0)

        return prepare_kernel(step, compile_mode)
    definition = E.resolve("constraint", spec)
    config = spec.get("config", {})

    def step(value):
        return definition.function(value, config)

    update = prepare_kernel(step, compile_mode)

    def checked(value):
        return X.checked_tensor(
            update(value),
            shape=value.shape,
            device=value.device,
            dtype=value.dtype,
            name=definition.name,
            check_values=check_values,
        )

    return checked


def operation_kernel(operation, *, compile_mode):
    spec = copy.deepcopy(operation["config"])
    config = spec.get("settings", {})
    definition = E.resolve(
        "operation", {"definition": spec["definition"], "config": config}
    )

    def step(sources, parameters):
        return definition.function(sources, parameters, config)

    return definition, prepare_kernel(step, compile_mode)
