"""Tensor-only local learning kernels, using pre-update decayed spike traces."""

import math

from snnlab.plasticity import InhibitorySTDP, PairSTDP, RewardModulatedSTDP, TripletSTDP


def initial_state(rule, weight):
    state = {
        "pre": weight.new_zeros((1, weight.shape[0])),
        "post": weight.new_zeros((1, weight.shape[1])),
    }
    if isinstance(rule, TripletSTDP):
        state.update(pre_slow=state["pre"].clone(), post_slow=state["post"].clone())
    if isinstance(rule, RewardModulatedSTDP):
        state["eligibility"] = weight.new_zeros(weight.shape)
    return state


def _pair(rule, weight, pre, post, old_pre, old_post):
    potentiation = rule.a_plus.value * (old_pre.T @ post)
    depression = rule.a_minus.value * (pre.T @ old_post)
    if rule.weight_dependence == "multiplicative":
        span = rule.w_max.value - rule.w_min.value
        potentiation = potentiation * ((rule.w_max.value - weight) / span)
        depression = depression * ((weight - rule.w_min.value) / span)
    return potentiation - depression


def step(rule, weight, pre, post, state, dt_ms, reward=None):
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
