"""Small, deterministic authoring cases for manual diagram review."""

import json
from dataclasses import dataclass, replace
from pathlib import Path

from snnlab import lang, viz
from snnlab.lang.compilation.bundles import Bundle


@dataclass(frozen=True)
class VisualCase:
    name: str
    description: str
    bundle: Bundle
    external_encoder: bool = False


def representative_classifier() -> VisualCase:
    net = lang.Network("two_layer_classifier", dt=0.1 * lang.ms)
    inputs = net.input(
        "inputs", shape=("time", "batch", 32), signal_type="spikes", unit="spike"
    )
    source = inputs
    for name, size in (("layer_1", 64), ("layer_2", 32)):
        with net.group(name):
            excitatory = net.population(
                f"{name}_E", size=size, neuron=lang.COBA_LIF(tau_mem=20 * lang.ms)
            )
            inhibitory = net.population(
                f"{name}_I",
                size=size // 4,
                neuron=lang.COBA_LIF(tau_mem=5 * lang.ms),
            )
            net.connect(
                excitatory.spikes,
                inhibitory.excitatory,
                name=f"{name}_E_to_I",
                synapse=lang.AMPA(tau=2 * lang.ms),
                weight=lang.Constant(0.2),
                constraint=lang.NonNegative(),
                connection="recurrent",
                delay=0.1 * lang.ms,
            )
            net.connect(
                inhibitory.spikes,
                excitatory.inhibitory,
                name=f"{name}_I_to_E",
                synapse=lang.GABA(tau=9 * lang.ms),
                weight=lang.Constant(0.5),
                constraint=lang.NonNegative(),
                connection="feedback",
                delay=0.1 * lang.ms,
            )
        net.connect(
            source,
            excitatory.excitatory,
            name=f"input_to_{name}",
            synapse=lang.AMPA(tau=2 * lang.ms),
            weight=lang.Constant(0.1),
            constraint=lang.NonNegative(),
        )
        source = excitatory.spikes

    scores = lang.SpikeCount(source=source, classes=4, name="classifier")
    net.output("class_scores", scores)
    training = lang.TrainSpec(
        objectives=[lang.CrossEntropy(prediction=scores, target="label")],
        parameter_groups=[
            lang.ParameterGroup(
                [parameter["id"] for parameter in net.parameters], name="all", lr=1e-3
            )
        ],
        optimizer=lang.AdamW(),
    )
    return VisualCase(
        name="two_layer_classifier",
        description=(
            "32 spike inputs feed two grouped E/I layers (64/16 and 32/8 neurons), "
            "with excitatory recurrence, inhibitory feedback, a four-class spike-count "
            "readout, and a cross-entropy training recipe using AdamW."
        ),
        bundle=lang.compile(net, training=training),
    )


def all_box_types() -> VisualCase:
    net = lang.Network("all_box_types", dt=0.1 * lang.ms)
    inputs = net.input(
        "inputs", shape=("time", "batch", 16), signal_type="spikes", unit="spike"
    )
    with net.group("circuit"):
        e = net.population(
            "circuit_E", size=16, neuron=lang.COBA_LIF(tau_mem=20 * lang.ms)
        )
        i = net.population(
            "circuit_I", size=4, neuron=lang.COBA_LIF(tau_mem=5 * lang.ms)
        )
        forward = net.connect(
            e.spikes,
            i.excitatory,
            name="E_to_I",
            synapse=lang.AMPA(),
            weight=lang.Constant(0.2),
            connection="recurrent",
            delay=0.1 * lang.ms,
        )
        feedback = net.connect(
            i.spikes,
            e.inhibitory,
            name="I_to_E",
            synapse=lang.GABA(),
            weight=lang.Constant(0.5),
            connection="feedback",
            delay=0.1 * lang.ms,
        )
    drive = net.connect(
        inputs,
        e.excitatory,
        name="input_projection",
        synapse=lang.AMPA(),
        weight=lang.Constant(0.1),
    )
    readout = net.population(
        "readout", size=4, spiking=False, neuron=lang.LeakyIntegrator(tau=20 * lang.ms)
    )
    projection = net.connect(
        e.spikes,
        readout.excitatory,
        name="readout_projection",
        synapse=lang.LeakyIntegrator(tau=10 * lang.ms),
        weight=lang.Normal(0.1, 0.01),
    )
    scores = lang.reduce(
        readout.voltage, operation="mean", over="time", name="mean_scores"
    )
    net.output("class_scores", scores)
    training = lang.TrainSpec(
        objectives=[lang.CrossEntropy(prediction=scores, target="label")],
        parameter_groups=[
            lang.ParameterGroup(
                [drive.weight, projection.weight], name="learned", lr=0.001
            ),
            lang.ParameterGroup(
                [forward.weight, feedback.weight],
                name="fixed_recurrence",
                lr=0,
                frozen=True,
            ),
        ],
        optimizer=lang.AdamW(),
    )
    return VisualCase(
        name="all_box_types",
        description="All six authoring box types plus a neutral external-encoder annotation: inputs, spiking and analogue populations, a mean operation, output, trainable/frozen parameter groups, and a cross-entropy objective. The E/I group is always explicit. The external encoder illustrates a manual diagram annotation, outside the compiled network.",
        bundle=lang.compile(net, training=training),
        external_encoder=True,
    )


def case_diagram(case: VisualCase, view: str) -> viz.Diagram:
    visual = lang.diagram(case.bundle, view=view)
    if case.external_encoder:
        visual = replace(
            visual,
            nodes=(
                viz.DiagramNode(
                    "external_encoder",
                    "External encoder",
                    "Samples to spike trains\nOutside compiled network",
                    "annotation",
                    kind="neutral",
                    classes=("neutral",),
                ),
                *visual.nodes,
            ),
            edges=(
                viz.DiagramEdge("external_encoder", "inputs", role="signal"),
                *visual.edges,
            ),
        )
    return visual


def cases() -> tuple[VisualCase, ...]:
    descriptions = {
        "exp124_coba": "Ungrouped two-layer COBA classifier with analogue readout, reductions, score normalization, and trainable/frozen parameter groups.",
        "exp124_ping": "The same classifier with two local E/I recurrent loops; exposes ungrouped population placement and training annotations.",
        "exp131_coupled": "Two independently driven 80-E/20-I PING circuits with reciprocal E-to-E and E-to-I coupling.",
        "exp131_phase_response": "One PING circuit with separate drive, E probe, and I probe inputs.",
        "exp133_coupled": "Two PING circuits with 1,024-channel drives and reciprocal, delayed, step-modulated coupling.",
        "exp133_sender_receiver": "Independently driven sender and receiver PING circuits with one-way, delayed E-to-E and E-to-I coupling.",
    }
    snapshots = []
    for name, description in descriptions.items():
        payload = json.loads(
            (Path(__file__).with_name("inputs") / f"{name}.json").read_text()
        )
        bundle = Bundle(
            graph=payload["graph"],
            training=payload["training"],
            manifest=payload["manifest"],
            diagnostics=[],
        )
        snapshots.append(
            VisualCase(
                name=name,
                description=description
                + " Frozen authoring snapshot from "
                + payload["source"]["path"]
                + ".",
                bundle=bundle,
            )
        )
    return (all_box_types(), representative_classifier(), *snapshots)
