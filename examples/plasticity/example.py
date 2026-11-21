import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from snnlab import extensions, lang, sim, viz
from snnlab.plasticity import (
    CustomPlasticity,
    InhibitorySTDP,
    PairSTDP,
    Plasticity,
    RewardModulatedSTDP,
    TripletSTDP,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def prepare_results():
    if OUTPUT_DIR.is_symlink():
        OUTPUT_DIR.unlink()
    elif OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)


def initialize_hebbian(ctx):
    return {"coincidences": ctx.weights.new_zeros(ctx.weights.shape)}


def hebbian_step(ctx):
    coincidences = ctx.pre_spikes.T @ ctx.post_spikes
    return ctx.config["amplitude"] * coincidences, {
        "coincidences": ctx.state["coincidences"] + coincidences
    }


def make_graph(*, inhibitory=False, reward=False):
    net = lang.Network("plasticity", dt=1 * lang.ms)
    events = net.input(
        "events",
        shape=("time", "batch", 2),
        signal_type="spikes",
        unit="spike",
    )
    if reward:
        net.input(
            "reward",
            shape=("time", "batch", 1),
            signal_type="analog",
            unit="1",
        )
    cells = net.population(
        "cells",
        size=2,
        neuron=lang.LIF(
            resting_mv=0,
            reset_mv=0,
            initial_voltage_mv=0,
            threshold_mv=0.8,
            tau_mem=20 * lang.ms,
        ),
    )
    net.connect(
        events,
        cells.inhibitory if inhibitory else cells.excitatory,
        name="drive",
        synapse=lang.ExponentialCurrent(tau=2 * lang.ms),
        weight=lang.Constant(0.3),
        initialization_scaling="direct",
        constraint=lang.NonNegative(),
    )
    if inhibitory:
        net.connect(
            events,
            cells.excitatory,
            name="fixed_excitation",
            synapse=lang.ExponentialCurrent(tau=2 * lang.ms),
            weight=lang.Constant(0.8),
            initialization_scaling="direct",
        )
    net.output("spikes", cells.spikes)
    return lang.compile(net, target=None)


def learning_rules(definition):
    pair = dict(
        a_plus=0.008 * lang.nA,
        a_minus=0.004 * lang.nA,
        tau_pre=10 * lang.ms,
        tau_post=10 * lang.ms,
        w_min=0 * lang.nA,
        w_max=0.8 * lang.nA,
    )
    rules = {
        "STDP": PairSTDP(**pair),
        "Weight-dependent": PairSTDP(**pair, weight_dependence="multiplicative"),
        "Triplet": TripletSTDP(
            a2_plus=0.002 * lang.nA,
            a2_minus=0.004 * lang.nA,
            a3_plus=0.006 * lang.nA,
            a3_minus=0 * lang.nA,
            tau_pre=10 * lang.ms,
            tau_post=20 * lang.ms,
            tau_x=40 * lang.ms,
            tau_y=40 * lang.ms,
            w_min=0 * lang.nA,
            w_max=0.8 * lang.nA,
        ),
        "Inhibitory": InhibitorySTDP(
            eta=0.005 * lang.nA,
            target_rate=20 * lang.Hz,
            tau=20 * lang.ms,
            w_min=0 * lang.nA,
            w_max=0.8 * lang.nA,
        ),
        "Reward-modulated": RewardModulatedSTDP(
            **pair,
            reward_input="reward",
            tau_eligibility=50 * lang.ms,
            tau_weight=20 * lang.ms,
        ),
        "Custom Hebbian": CustomPlasticity(
            definition=definition,
            config={"amplitude": 0.005},
            w_min=0 * lang.nA,
            w_max=0.8 * lang.nA,
        ),
    }
    return {name: Plasticity(rules={"drive": rule}) for name, rule in rules.items()}


def simulate_rule(request, events, rewards, stem):
    rule = request.rules["drive"]
    bundle = make_graph(
        inhibitory=isinstance(rule, InhibitorySTDP),
        reward=isinstance(rule, RewardModulatedSTDP),
    )
    viz.render_diagram(
        lang.diagram(bundle),
        OUTPUT_DIR / ("network.png" if stem == "stdp" else f"{stem}-network.png"),
    )
    state = None
    weights = [np.full((2, 2), 0.3)]
    outputs = []
    for start in range(0, len(events), 10):
        bindings = [sim.DenseArrayBinding("events", events[start : start + 10])]
        if isinstance(rule, RewardModulatedSTDP):
            bindings.append(
                sim.DenseArrayBinding("reward", rewards[start : start + 10])
            )
        result = sim.simulate(
            graph=bundle.graph,
            device="cpu",
            diagnostics=False,
            input_bindings=bindings,
            plasticity=request,
            runtime_state=state,
        )
        state = result.runtime_state
        weights.append(result.parameters["drive.weight"].numpy())
        outputs.append(result.outputs["spikes"].numpy())
    history = np.stack(weights)
    np.savez(
        OUTPUT_DIR / f"{stem}.npz",
        weights=history,
        spikes=np.concatenate(outputs),
        input_spikes=events.numpy(),
        time_ms=np.arange(0, 201, 10),
        reward=rewards.numpy()
        if isinstance(rule, RewardModulatedSTDP)
        else np.empty(0),
    )
    sim.save_runtime_state(OUTPUT_DIR / f"{stem}-state", state)
    (OUTPUT_DIR / f"{stem}-rule.json").write_text(
        json.dumps(request.json(), indent=2) + "\n"
    )
    return history


def main():
    prepare_results()
    definition = "examples.local_hebbian/v1"
    extensions.register_plasticity(
        definition,
        "example:hebbian_step",
        initialize="example:initialize_hebbian",
        state_units={"coincidences": "1"},
    )
    events = torch.zeros(200, 1, 2)
    events[::5, 0, 0] = 1
    events[2::7, 0, 1] = 1
    rewards = torch.zeros(200, 1, 1)
    rewards[60:100] = 1
    rewards[140:180] = -1
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for (label, request), ax in zip(
        learning_rules(definition).items(), axes.flat, strict=True
    ):
        stem = {"STDP": "stdp", "Custom Hebbian": "custom"}.get(label, label.lower())
        history = simulate_rule(request, events, rewards, stem)
        for source in range(2):
            for target in range(2):
                ax.plot(
                    np.arange(0, 201, 10),
                    history[:, source, target],
                    label=f"{source} → {target}",
                )
        if isinstance(request.rules["drive"], RewardModulatedSTDP):
            ax.axvspan(60, 100, color="green", alpha=0.1)
            ax.axvspan(140, 180, color="red", alpha=0.1)
        ax.set(title=label, xlabel="Time (ms)", ylabel="Weight (nA)", ylim=(0, 0.85))
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "plasticity.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
