"""Stationary noisy-LIF gain with an analytic current derivative."""

from dataclasses import dataclass

import numpy as np
from scipy.special import erfcx

from snnlab.meanfield.model import magnitude


@dataclass(frozen=True)
class NoisyLIF:
    sigma: object
    quadrature_nodes: int = 48

    def __post_init__(self):
        if magnitude(self.sigma, "mV") <= 0:
            raise ValueError("sigma must be positive")
        if not isinstance(self.quadrature_nodes, int) or self.quadrature_nodes < 8:
            raise ValueError("quadrature_nodes must be an integer >= 8")

    def bind(self, ctx, population):
        neuron = population["neuron"]
        sigma = magnitude(self.sigma, "mV")
        tau = neuron.get("tau_mem", {"value": 20.0})["value"]
        capacitance = float(neuron.get("capacitance_nf", 1.0 if tau >= 15 else 0.5))
        leak = float(neuron.get("leak_us", capacitance / tau))
        tau = capacitance / leak if leak else 0.0
        rest = float(neuron.get("resting_mv", -65))
        reset = float(neuron.get("reset_mv", -65))
        threshold = float(neuron.get("threshold_mv", -50))
        refractory = neuron["refractory"]["value"]
        if (
            not np.isfinite([leak, tau, refractory, rest, reset, threshold]).all()
            or leak <= 0
            or tau <= 0
            or refractory <= 0
            or reset >= threshold
        ):
            raise ValueError(
                "NoisyLIF requires positive leak, tau, refractory and reset < threshold"
            )
        nodes, weights = np.polynomial.legendre.leggauss(self.quadrature_nodes)

        def evaluate(current):
            mean = rest + float(current) / leak
            a, b = (reset - mean) / sigma, (threshold - mean) / sigma
            if b > 26:
                return 0.0, 0.0
            points = (a + b) / 2 + (b - a) / 2 * nodes
            integral = (b - a) / 2 * np.dot(weights, erfcx(-points))
            rate = 1 / (refractory + tau * np.sqrt(np.pi) * integral)
            slope = (
                tau
                * np.sqrt(np.pi)
                * rate**2
                * (erfcx(-b) - erfcx(-a))
                / (leak * sigma)
            )
            return 1000 * rate, 1000 * slope

        ctx.assumptions.append(
            f"{population['id']}: fixed voltage-noise scale {sigma:g} mV; "
            "stationary white-noise LIF gain (noise is not derived from connectivity)"
        )
        gain = ctx.function(
            f"Phi_{population['id']}",
            lambda current: evaluate(current)[0],
            derivative=lambda current: evaluate(current)[1],
            definition={
                "kind": "noisy_lif",
                "sigma_mv": sigma,
                "tau_mem_ms": tau,
                "leak_us": leak,
                "resting_mv": rest,
                "reset_mv": reset,
                "threshold_mv": threshold,
                "refractory_ms": refractory,
                "quadrature_nodes": self.quadrature_nodes,
                "zero_rate_upper_bound": 26,
                "input_unit": "nA",
                "output_unit": "Hz",
            },
        )
        return gain, 1000 / refractory
