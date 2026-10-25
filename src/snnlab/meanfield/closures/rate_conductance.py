"""Homogeneous rate/conductance closure used by the Pinglab examples."""

from dataclasses import dataclass
from typing import Mapping

import sympy as sp

from snnlab.meanfield.model import Filter, magnitude


@dataclass(frozen=True)
class RateConductance:
    gain: object
    tau_rate: object
    driving_force: str = "resting"

    def build(self, ctx):
        if self.driving_force != "resting":
            raise ValueError("only fixed resting driving forces are supported")
        if ctx.graph.get("operations"):
            raise ValueError("RateConductance does not support graph operations")
        populations = {p["id"]: p for p in ctx.populations}
        if set(ctx.applied_current) - set(populations):
            raise ValueError("applied current references an unknown population")
        if isinstance(self.tau_rate, Mapping) and set(self.tau_rate) != set(
            populations
        ):
            raise ValueError("tau_rate must specify every population exactly once")
        currents, rates, gains, relaxation = {}, {}, {}, {}
        display_currents, driving_forces = {}, {}
        for name, population in populations.items():
            if population["neuron"]["kind"] != "coba_lif" or not population["spiking"]:
                raise ValueError(
                    f"{name}: RateConductance supports spiking COBA_LIF only"
                )
            gains[name], maximum = self.gain.bind(ctx, population)
            rates[name] = ctx.state(f"{name}.rate", unit="Hz", bounds=(0, maximum))
            tau = (
                self.tau_rate[name]
                if isinstance(self.tau_rate, Mapping)
                else self.tau_rate
            )
            if magnitude(tau, "ms") <= 0:
                raise ValueError("rate relaxation times must be positive")
            relaxation[name] = ctx.parameter(
                f"{name}.tau_rate", tau, unit="ms", bounds=(1e-300, float("inf"))
            )
            currents[name] = ctx.parameter(
                f"{name}.applied_current", ctx.applied_current.get(name, 0), unit="nA"
            )
            ctx.notation[rates[name]] = rf"r_{{{name}}}"
            ctx.notation[relaxation[name]] = rf"\tau_{{r,{name}}}"
            ctx.notation[currents[name]] = rf"I_{{\mathrm{{ext}},{name}}}"
            ctx.describe(
                rates[name], description=f"Mean firing rate of population {name}"
            )
            ctx.describe(
                relaxation[name],
                description=f"Rate relaxation time of population {name}",
            )
            ctx.describe(
                currents[name],
                description=f"Applied external current to population {name}",
            )
            display_currents[name] = currents[name]
            neuron = population["neuron"]
            rest = neuron.get("resting_mv", -65)
            driving_forces[name] = {
                "excitatory": ctx.display_definition(
                    f"{name}.driving_force_exc",
                    neuron.get("excitatory_reversal_mv", 0) - rest,
                    latex=rf"\Delta V_{{\mathrm{{exc}},{name}}}",
                    description=f"Excitatory reversal potential minus resting voltage of {name}",
                    unit="mV",
                ),
                "inhibitory": ctx.display_definition(
                    f"{name}.driving_force_inh",
                    rest - neuron.get("inhibitory_reversal_mv", -80),
                    latex=rf"\Delta V_{{\mathrm{{inh}},{name}}}",
                    description=f"Resting voltage of {name} minus inhibitory reversal potential",
                    unit="mV",
                ),
            }
        source_rates = {}
        for name, rate in ctx.inputs.items():
            source_rates[name] = ctx.parameter(
                f"{name}.rate", rate.rate, unit="Hz", bounds=(0, float("inf"))
            )
            ctx.notation[source_rates[name]] = rf"r_{{{name}}}"
            ctx.describe(
                source_rates[name],
                description=f"Mean firing rate per external source channel in {name}",
            )
        for projection in ctx.projections:
            name = projection["id"]
            delay_spec = projection.get("delay") or {"value": 0, "unit": "ms"}
            if delay_spec.get("unit") != "ms":
                raise ValueError(f"{name}: delay must be in ms")
            delay = magnitude(delay_spec["value"], "ms")
            if delay < 0:
                raise ValueError(f"{name}: delays must be nonnegative")
            ctx.projection_delays[name] = delay
            source, port = projection["source"].rsplit(".", 1)
            target = projection["target"].rsplit(".", 1)[0]
            synapse = projection["synapse"]
            if synapse["kind"] not in ("ampa", "gaba"):
                raise ValueError(f"{name}: requires an AMPA or GABA synapse")
            polarity = projection["target"].rsplit(".", 1)[1]
            expected = "excitatory" if synapse["kind"] == "ampa" else "inhibitory"
            if polarity != expected or projection.get("modulation"):
                raise ValueError(
                    f"{name}: incompatible polarity or graph modulation; supply explicit controls"
                )
            if source in populations and port != "spikes":
                raise ValueError(f"{name}: recurrent source must be spikes")
            if source not in populations and (
                source not in source_rates or port != "value"
            ):
                raise ValueError(f"{name}: missing external rate or unsupported source")
            tau_value = synapse.get("tau", {"value": 1, "unit": "ms"})["value"]
            if tau_value <= 0:
                raise ValueError("synaptic decay times must be positive")
            tau = ctx.parameter(
                f"{name}.tau", tau_value, unit="ms", bounds=(1e-300, float("inf"))
            )
            coupling = ctx.parameter(
                f"{name}.coupling",
                ctx.weights[name],
                unit="uS",
                bounds=(0, float("inf")),
            )
            gain = ctx.parameter(f"{name}.gain", 1, unit="1", bounds=(0, float("inf")))
            pathway = rf"{source}\to {target}"
            ctx.notation[tau] = rf"\tau_{{{pathway}}}"
            ctx.notation[coupling] = rf"C_{{{pathway}}}"
            ctx.notation[gain] = rf"a_{{{pathway}}}"
            source_count = (
                populations[source]["size"]
                if source in populations
                else next(
                    i["shape"][-1] for i in ctx.graph["inputs"] if i["id"] == source
                )
            )
            target_count = populations[target]["size"]
            ctx.describe(
                tau,
                description=f"Synaptic conductance decay time for {source} → {target}",
            )
            ctx.describe(
                gain,
                description=f"Dimensionless weight multiplier for {source} → {target}",
            )
            ctx.describe(
                coupling,
                description="Total effective incoming conductance jump summed over sources and averaged over target neurons, including zero edges",
                definition_latex=(
                    rf"C_{{{pathway}}} = \frac{{1}}{{N_{{{target}}}}}"
                    rf"\sum_{{j,k}}\mathbf{{W}}_{{jk}}^{{{pathway}}}"
                ),
                aggregation={
                    "operation": "sum_effective_weights_divided_by_target_count",
                    "projection": name,
                    "source": source,
                    "target": target,
                    "source_count": source_count,
                    "target_count": target_count,
                    "includes_zero_edges": True,
                    "allow_autapses": projection.get("allow_autapses"),
                },
                components={
                    f"N_{target}": {
                        "latex": rf"N_{{{target}}}",
                        "description": "Number of target neurons",
                        "value": target_count,
                        "unit": "1",
                    },
                    "W": {
                        "latex": rf"\mathbf{{W}}_{{jk}}^{{{pathway}}}",
                        "description": "Effective conductance jump per source spike; absent connections have zero weight",
                        "unit": "uS",
                        "indices": {
                            "j": {
                                "description": "Source neuron or input channel",
                                "size": source_count,
                            },
                            "k": {"description": "Target neuron", "size": target_count},
                        },
                    },
                },
            )
            recurrent = source in populations
            if (
                ctx.delays == "explicit"
                and delay
                and not recurrent
                and ctx.external_filters == "quasistatic"
            ):
                raise ValueError(
                    "explicit external delays require dynamic external filters"
                )
            receptor = "e" if expected == "excitatory" else "i"
            conductance_label = rf"g_{{{receptor}}}^{{{target}}}"
            matching = [
                p for p in ctx.projections if p["target"] == projection["target"]
            ]
            if recurrent:
                if (
                    sum(p["source"].rsplit(".", 1)[0] in populations for p in matching)
                    > 1
                ):
                    conductance_label = rf"g_{{{receptor},{source}}}^{{{target}}}"
            else:
                external_count = sum(
                    p["source"].rsplit(".", 1)[0] not in populations for p in matching
                )
                conductance_label = (
                    rf"g_{{\mathrm{{ext}}}}^{{{target}}}"
                    if external_count == 1
                    else rf"g_{{\mathrm{{ext}},{source}}}^{{{target}}}"
                )
            if recurrent or ctx.external_filters == "dynamic":
                conductance = ctx.state(
                    f"{name}.conductance", unit="uS", bounds=(0, float("inf"))
                )
                rate = rates[source] if recurrent else source_rates[source]
                if ctx.delays == "explicit":
                    rate = ctx.delay(rate, delay, name=f"{name}.delayed_rate")
                ctx.derivative(
                    conductance, -conductance / tau + coupling * gain * rate / 1000
                )
                if recurrent:
                    ctx.filters.append(
                        Filter(conductance, source, tau, coupling * gain)
                    )
                ctx.notation[conductance] = conductance_label
                ctx.describe(
                    conductance,
                    description=f"Mean synaptic conductance from {source} into {target}",
                )
                display_conductance = conductance
            else:
                conductance = tau * coupling * gain * source_rates[source] / 1000
                display_conductance = ctx.display_definition(
                    f"{name}.mean_conductance",
                    conductance,
                    latex=conductance_label,
                    description=f"Quasistatic mean external conductance from {source} into {target}",
                    unit="uS",
                )
            neuron = populations[target]["neuron"]
            reversal = (
                neuron.get("excitatory_reversal_mv", 0)
                if expected == "excitatory"
                else neuron.get("inhibitory_reversal_mv", -80)
            )
            force = reversal - neuron.get("resting_mv", -65)
            currents[target] += force * conductance
            signed_force = driving_forces[target][expected]
            if expected == "inhibitory":
                signed_force = -signed_force
            display_currents[target] += signed_force * display_conductance
        for name in populations:
            ctx.derivative(
                rates[name],
                (gains[name](currents[name]) - rates[name]) / relaxation[name],
            )
            current = ctx.display_definition(
                f"{name}.effective_current",
                sp.collect(display_currents[name], list(driving_forces[name].values())),
                latex=rf"I_{{{name}}}",
                description=f"Effective input current to population {name} with fixed resting driving forces",
                unit="nA",
            )
            ctx.display_derivatives[rates[name]] = (
                gains[name](current) - rates[name]
            ) / relaxation[name]
        ctx.assumptions.extend(
            [
                "Homogeneous source rates; fixed weights summarised by mean incoming weight per target",
                "Fixed resting driving forces; shunting and voltage/conductance covariance omitted",
                "Input covariance and within-population weight heterogeneity omitted",
                f"External filters: {ctx.external_filters}; delays: {ctx.delays}",
                "Rates in Hz; time in ms; conductance in uS; current in nA",
                "Projection gain acts at target delivery, multiplying delayed source activity; continuous dynamical state is retained across switches",
            ]
        )
        return ctx.build()
