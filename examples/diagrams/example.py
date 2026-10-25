"""Render a gallery of SNNLang circuit and training diagrams."""

import argparse
import html
import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

from snnlab import lang, viz
from snnlab.lang.compilation.bundles import Bundle

OUTPUT_DIR = Path(__file__).resolve().parent / "results"


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


def case_diagram(case: VisualCase, view: str, *, weight_labels=False) -> viz.Diagram:
    visual = lang.diagram(case.bundle, view=view, weight_labels=weight_labels)
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
        "exp136_sustained_cue_ping": "Two independently driven sender PING circuits converge on a receiver, each with its own non-recurrent neuronal cue population. Five authored groups include sensory and cue inputs. Uncalibrated circuit proposal; no simulation or training results.",
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


def prepare_results(output=None):
    output = OUTPUT_DIR if output is None else output
    if output.is_symlink():
        output.unlink()
    elif output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--scale", type=int, default=2, help="PNG scale (default: 2)")
    args = parser.parse_args()
    if args.scale < 1:
        parser.error("--scale must be a positive integer")
    missing = [program for program in ("dot", "neato") if shutil.which(program) is None]
    if missing:
        parser.error(
            f"Graphviz executables must be available on PATH: {', '.join(missing)}"
        )

    output = args.out_dir.absolute()
    prepare_results(output)
    sections = []
    navigation = []
    for case in cases():
        navigation.append(
            f'<li><a href="#{case.name}">{html.escape(case.name)}</a></li>'
        )
        figures = []
        views = (
            ("training", "circuit")
            if case.name == "all_box_types"
            else ("circuit", "training")
        )
        if case.name in {"all_box_types", "two_layer_classifier", "exp131_coupled"}:
            views = (*views, "circuit_weights")
        for name in views:
            view = "circuit" if name == "circuit_weights" else name
            weight_labels = name == "circuit_weights"
            if weight_labels and case.name == "all_box_types":
                weight_labels = {
                    "input_projection": viz.WeightLabel(subscript="in"),
                    "E_to_I": viz.WeightLabel("J", "E→I", "rec"),
                    "I_to_E": viz.WeightLabel("J", "I→E", "rec"),
                    "readout_projection": viz.WeightLabel(subscript="out"),
                }
            elif weight_labels and case.name == "two_layer_classifier":
                weight_labels = {
                    "input_to_layer_1": viz.WeightLabel(subscript="in"),
                    "input_to_layer_2": viz.WeightLabel(subscript="E1→E2"),
                    **{
                        f"layer_{layer}_{path}": viz.WeightLabel(
                            subscript=f"{source}{layer}→{target}{layer}"
                        )
                        for layer in (1, 2)
                        for path, source, target in (
                            ("E_to_I", "E", "I"),
                            ("I_to_E", "I", "E"),
                        )
                    },
                }
            elif weight_labels and case.name == "exp131_coupled":
                aliases = {
                    "drive_A_300_Hz": "inA",
                    "drive_B_260_Hz": "inB",
                    "PING_A_E": "EA",
                    "PING_A_I": "IA",
                    "PING_B_E": "EB",
                    "PING_B_I": "IB",
                }
                weight_labels = {
                    row["id"]: viz.WeightLabel(
                        subscript=f"{aliases[row['source'].partition('.')[0]]}→{aliases[row['target'].partition('.')[0]]}"
                    )
                    for row in case.bundle.graph["projections"]
                }
            diagram = case_diagram(case, view, weight_labels=weight_labels)
            stem = f"{case.name}/{name}"
            for suffix in ("svg", "png"):
                viz.render_diagram(
                    diagram, output / f"{stem}.{suffix}", scale=args.scale
                )
            options = f"view={view}; weight labels={bool(weight_labels)}; all groups explicit; PNG scale={args.scale}"
            legend = ""
            if name == "circuit_weights":
                legend = {
                    "all_box_types": "W: projection weights; J: recurrent weights; E/I: excitatory/inhibitory populations; in/out: input/readout; rec: recurrent.",
                    "two_layer_classifier": "W: projection weights; E1/I1: excitatory/inhibitory populations in layer 1; E2/I2: populations in layer 2; in: input weights.",
                    "exp131_coupled": "W: projection weights; EA/IA: excitatory/inhibitory populations in PING A; EB/IB: populations in PING B; inA/inB: their external drives.",
                }[case.name] + " Subscripts use source→target order."
            figures.append(
                f"<figure><figcaption><h3>{html.escape(name.replace('_', ' '))}</h3>"
                + (f"<p>{html.escape(legend)}</p>" if legend else "")
                + f"<code>{html.escape(options)}</code> "
                f'<a href="{stem}.svg">SVG</a> · <a href="{stem}.png">PNG</a>'
                f'</figcaption><a href="{stem}.svg">'
                f'<img src="{stem}.svg" alt="{html.escape(case.name)}: {name}"></a></figure>'
            )
        sections.append(
            f'<section id="{case.name}"><h2>{html.escape(case.name)}</h2>'
            f"<p>{html.escape(case.description)}</p>{''.join(figures)}</section>"
        )
    page = (
        """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SNNLang diagram review</title>
<style>
body { margin: 2rem auto; padding: 0 1rem; max-width: 1400px;
       font: 16px/1.5 system-ui, sans-serif; color: #20252b; background: #f5f6f8; }
figure { margin: 1.5rem 0; padding: 1rem; background: white;
         border: 1px solid #ccd2d9; border-radius: 8px; overflow-x: auto; }
h3 { margin: 0 0 .5rem; text-transform: capitalize; }
code { display: block; margin-bottom: .5rem; overflow-wrap: anywhere; }
img { display: block; max-width: 100%; height: auto; margin: 1rem auto 0; }
a { color: #1858a8; }
</style></head><body><h1>SNNLang diagram review</h1>
<p>Inspect labels, group boundaries, arrow routing, overlaps, and training annotations.
Open an SVG for full-size inspection; PNGs are available alongside each view.</p>
"""
        + '<nav aria-label="Cases"><ol>'
        + "".join(navigation)
        + "</ol></nav>"
        + "".join(sections)
        + "</body></html>\n"
    )
    index = output / "index.html"
    index.write_text(page, encoding="utf-8")
    print(f"Gallery: {index}")


if __name__ == "__main__":
    main()
