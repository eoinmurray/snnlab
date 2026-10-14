"""Prepared tensor kernels; graph orchestration and callbacks remain eager."""

from __future__ import annotations

import copy
import math

import torch

from snnlab import _neurons as N
from snnlab.sim import extensions as X
from snnlab.sim import models as M


def neuron_kernel(population, *, dt_ms, refractory, spike_function, compile_mode=None):
    config = copy.deepcopy(population["neuron"])
    kind = config["kind"]
    ref_steps = refractory.get("refractory_steps", 0)
    if kind == "cuba_lif":

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
    if compile_mode is not None:
        # Each prepared closure owns its compiler guards. Sharing a code object
        # across unrelated neuron configurations exhausts Dynamo's cache limit.
        step.__code__ = step.__code__.replace(co_name=f"neuron_step_{id(step)}")
        step = torch.compile(step, mode=compile_mode, fullgraph=True, dynamic=False)
    return step
