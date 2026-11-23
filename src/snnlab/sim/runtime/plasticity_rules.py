"""Tensor-only local learning kernels, using pre-update decayed spike traces."""

import math

import torch

from snnlab.plasticity import (
    CalciumPlasticity,
    InhibitorySTDP,
    NonlinearSTDP,
    PairSTDP,
    RewardModulatedSTDP,
    SynapticScaling,
    TripletSTDP,
    VoltageSTDP,
)


def initial_state(rule, weight):
    if isinstance(rule, CalciumPlasticity):
        return {"calcium": weight.new_zeros(weight.shape)}
    if isinstance(rule, SynapticScaling):
        return {}
    state = {
        "pre": weight.new_zeros((1, weight.shape[0])),
        "post": weight.new_zeros((1, weight.shape[1])),
    }
    if isinstance(rule, TripletSTDP):
        state.update(pre_slow=state["pre"].clone(), post_slow=state["post"].clone())
    if isinstance(rule, RewardModulatedSTDP):
        state["eligibility"] = weight.new_zeros(weight.shape)
    if isinstance(rule, VoltageSTDP):
        state.pop("post")
        state.update(
            voltage_minus=weight.new_full(
                (1, weight.shape[1]), rule.voltage_rest.value
            ),
            voltage_plus=weight.new_full((1, weight.shape[1]), rule.voltage_rest.value),
        )
    return state


def _pair(rule, weight, pre, post, old_pre, old_post):
    potentiation = rule.a_plus.value * (old_pre.T @ post)
    depression = rule.a_minus.value * (pre.T @ old_post)
    if isinstance(rule, NonlinearSTDP) or rule.weight_dependence == "multiplicative":
        span = rule.w_max.value - rule.w_min.value
        mu_plus = rule.mu_plus if isinstance(rule, NonlinearSTDP) else 1
        mu_minus = rule.mu_minus if isinstance(rule, NonlinearSTDP) else 1
        potentiation = potentiation * ((rule.w_max.value - weight) / span).clamp(
            0, 1
        ).pow(mu_plus)
        depression = depression * ((weight - rule.w_min.value) / span).clamp(0, 1).pow(
            mu_minus
        )
    return potentiation - depression


def _calcium(rule, weight, pre, post, state, dt_ms):
    calcium = state["calcium"] * math.exp(-dt_ms / rule.tau_calcium.value)
    calcium = calcium + rule.c_pre * pre.T + rule.c_post * post
    span = rule.w_max.value - rule.w_min.value
    rho = (weight - rule.w_min.value) / span
    drift = rho * (1 - rho) * (rho - rule.rho_star)
    drift += rule.gamma_potentiation * (1 - rho) * (calcium > rule.theta_potentiation)
    drift -= rule.gamma_depression * rho * (calcium > rule.theta_depression)
    return span * dt_ms / rule.tau_weight.value * drift, {"calcium": calcium}


def _voltage(rule, pre, post, state, dt_ms, voltage):
    sample = torch.where(post.bool(), rule.spike_voltage.value, voltage)
    minus = sample + (state["voltage_minus"] - sample) * math.exp(
        -dt_ms / rule.tau_minus.value
    )
    plus = sample + (state["voltage_plus"] - sample) * math.exp(
        -dt_ms / rule.tau_plus.value
    )
    trace = state["pre"] * math.exp(-dt_ms / rule.tau_pre.value)
    ltd = rule.a_minus.value * (pre.T @ (minus - rule.theta_minus.value).clamp_min(0))
    gate = (sample - rule.theta_plus.value).clamp_min(0) * (
        plus - rule.theta_minus.value
    ).clamp_min(0)
    ltp = rule.a_plus.value * dt_ms * (trace.T @ gate)
    return ltp - ltd, {
        "pre": trace + pre,
        "voltage_minus": minus,
        "voltage_plus": plus,
    }


def scaling_step(rule, weight, post, rate, integral, dt_ms):
    dt_s = dt_ms / 1000
    rate = (
        rate * math.exp(-dt_ms / rule.tau_rate.value)
        + post * 1000 / rule.tau_rate.value
    )
    error = rule.target_rate.value - rate
    integral = integral + dt_s * error
    delta = weight * dt_s * (rule.beta * error + rule.gamma * integral)
    return delta, rate, integral


def step(rule, weight, pre, post, state, dt_ms, reward=None, voltage=None):
    if isinstance(rule, CalciumPlasticity):
        return _calcium(rule, weight, pre, post, state, dt_ms)
    if isinstance(rule, VoltageSTDP):
        if voltage is None:
            raise ValueError("VoltageSTDP requires postsynaptic voltage samples")
        return _voltage(rule, pre, post, state, dt_ms, voltage)
    if isinstance(rule, SynapticScaling):
        return torch.zeros_like(weight), {}
    tau_pre = rule.tau if isinstance(rule, InhibitorySTDP) else rule.tau_pre
    tau_post = rule.tau if isinstance(rule, InhibitorySTDP) else rule.tau_post
    old_pre = state["pre"] * math.exp(-dt_ms / tau_pre.value)
    old_post = state["post"] * math.exp(-dt_ms / tau_post.value)
    updated = {"pre": old_pre + pre, "post": old_post + post}
    if isinstance(rule, TripletSTDP):
        pre_slow = state["pre_slow"] * math.exp(-dt_ms / rule.tau_x.value)
        post_slow = state["post_slow"] * math.exp(-dt_ms / rule.tau_y.value)
        delta = old_pre.T @ (
            post * (rule.a2_plus.value + rule.a3_plus.value * post_slow)
        )
        delta -= (
            pre * (rule.a2_minus.value + rule.a3_minus.value * pre_slow)
        ).T @ old_post
        updated.update(pre_slow=pre_slow + pre, post_slow=post_slow + post)
    elif isinstance(rule, InhibitorySTDP):
        alpha = 2 * rule.tau.value * rule.target_rate.value / 1000
        delta = rule.eta.value * (pre.T @ (old_post - alpha) + old_pre.T @ post)
    elif isinstance(rule, PairSTDP):
        delta = _pair(rule, weight, pre, post, old_pre, old_post)
        if isinstance(rule, RewardModulatedSTDP):
            eligibility = (
                state["eligibility"] * math.exp(-dt_ms / rule.tau_eligibility.value)
                + delta
            )
            updated["eligibility"] = eligibility
            delta = (dt_ms / rule.tau_weight.value) * reward * eligibility
    else:
        raise TypeError("unsupported built-in plasticity rule")
    return delta, updated
