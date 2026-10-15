"""Built-in and registered neuron and synapse declarations."""

from __future__ import annotations

from typing import Any

from snnlab.lang._specs import Spec, _custom
from snnlab.lang.units import Quantity, ms


def COBA_LIF(*, refractory=None, **values: Any) -> Spec:
    _check_refractory_arguments(values)
    if refractory is None:
        tau = values.get("tau_mem", 20 * ms)
        refractory = (
            3.0 if isinstance(tau, Quantity) and tau.value >= 15 else 1.5
        ) * ms
    return Spec("coba_lif", dict(values, refractory=refractory))


def _check_refractory_arguments(values: dict[str, Any]) -> None:
    for key in ("refractory_steps", "refractory_policy"):
        if key in values:
            raise TypeError(
                f"{key} is no longer supported; use refractory=duration * ms "
                "with an exactly compatible graph timestep"
            )


def LIF(**values: Any) -> Spec:
    return CUBA_LIF(**values)


def CUBA_LIF(
    *,
    tau_mem=20 * ms,
    capacitance_nf=1.0,
    resting_mv=-65.0,
    threshold_mv=-50.0,
    reset_mv=-65.0,
    refractory=0 * ms,
    initial_voltage_mv=None,
    voltage_grad_dampen=1.0,
) -> Spec:
    return Spec(
        "cuba_lif",
        dict(
            tau_mem=tau_mem,
            capacitance_nf=capacitance_nf,
            resting_mv=resting_mv,
            threshold_mv=threshold_mv,
            reset_mv=reset_mv,
            refractory=refractory,
            initial_voltage_mv=resting_mv
            if initial_voltage_mv is None
            else initial_voltage_mv,
            voltage_grad_dampen=voltage_grad_dampen,
        ),
    )


def ExponentialCurrent(*, tau=5 * ms) -> Spec:
    return Spec("exponential_current", {"tau": tau})


def ALIF(**values: Any) -> Spec:
    """Current-based LIF with a spike-triggered adapting threshold."""
    return CUBA_ALIF(**values)


def CUBA_ALIF(
    *,
    tau_adaptation=200 * ms,
    adaptation_increment_mv=1.0,
    initial_adaptation_mv=0.0,
    **lif_values: Any,
) -> Spec:
    return Spec(
        "cuba_alif",
        dict(
            CUBA_LIF(**lif_values).values,
            tau_adaptation=tau_adaptation,
            adaptation_increment_mv=adaptation_increment_mv,
            initial_adaptation_mv=initial_adaptation_mv,
        ),
    )


def COBA_ALIF(
    *, excitatory_reversal_mv=0.0, inhibitory_reversal_mv=-80.0, **values: Any
) -> Spec:
    return Spec(
        "coba_alif",
        dict(
            CUBA_ALIF(**values).values,
            excitatory_reversal_mv=excitatory_reversal_mv,
            inhibitory_reversal_mv=inhibitory_reversal_mv,
        ),
    )


def ADEX(**values: Any) -> Spec:
    """Current-based adaptive exponential integrate-and-fire neuron."""
    return CUBA_ADEX(**values)


def CUBA_ADEX(
    *,
    tau_adaptation=200 * ms,
    delta_t_mv=2.0,
    spike_mv=-30.0,
    a_us=0.002,
    b_na=0.05,
    initial_adaptation_na=0.0,
    **lif_values: Any,
) -> Spec:
    return Spec(
        "cuba_adex",
        dict(
            CUBA_LIF(**lif_values).values,
            tau_adaptation=tau_adaptation,
            delta_t_mv=delta_t_mv,
            spike_mv=spike_mv,
            a_us=a_us,
            b_na=b_na,
            initial_adaptation_na=initial_adaptation_na,
        ),
    )


def COBA_ADEX(
    *, excitatory_reversal_mv=0.0, inhibitory_reversal_mv=-80.0, **values: Any
) -> Spec:
    return Spec(
        "coba_adex",
        dict(
            CUBA_ADEX(**values).values,
            excitatory_reversal_mv=excitatory_reversal_mv,
            inhibitory_reversal_mv=inhibitory_reversal_mv,
        ),
    )


def CustomNeuron(definition: str, **config: Any) -> Spec:
    return _custom("neuron", definition, config)


def CustomSynapse(definition: str, **config: Any) -> Spec:
    return _custom("synapse", definition, config)


def LeakyIntegrator(**values: Any) -> Spec:
    return Spec("leaky_integrator", values)


def AMPA(**values: Any) -> Spec:
    return Spec("ampa", values)


def GABA(**values: Any) -> Spec:
    return Spec("gaba", values)


def Modulatory(**values: Any) -> Spec:
    return Spec("modulatory", values)
