"""Export an editable E/I circuit and render its saved Excalidraw document."""

import shutil
from pathlib import Path

from snnlab import lang, viz

OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def author_network():
    net = lang.Network("editable_circuit", dt=0.1 * lang.ms)
    drive = net.input(
        "drive",
        shape=("time", "batch", 80),
        signal_type="spikes",
        unit="spike",
    )
    with net.group("circuit"):
        excitatory = net.population(
            "E",
            size=80,
            neuron=lang.COBA_LIF(tau_mem=20 * lang.ms),
        )
        inhibitory = net.population(
            "I",
            size=20,
            neuron=lang.COBA_LIF(tau_mem=5 * lang.ms),
        )
        net.connect(
            excitatory.spikes,
            inhibitory.excitatory,
            name="E_to_I",
            synapse=lang.AMPA(tau=2 * lang.ms),
            weight=lang.Constant(0.2),
            connection="recurrent",
            delay=0.1 * lang.ms,
        )
        net.connect(
            inhibitory.spikes,
            excitatory.inhibitory,
            name="I_to_E",
            synapse=lang.GABA(tau=9 * lang.ms),
            weight=lang.Constant(0.5),
            connection="feedback",
            delay=0.1 * lang.ms,
        )
    net.connect(
        drive,
        excitatory.excitatory,
        name="drive_to_E",
        synapse=lang.AMPA(tau=2 * lang.ms),
        weight=lang.Constant(0.1),
    )
    net.output("E_spikes", excitatory.spikes)
    return lang.compile(net)


def main():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)
    bundle = author_network()
    layout = viz.DiagramLayout(
        below={"I": "E"},
        align_y=[("drive", "E", "out:E_spikes")],
        edges={
            "drive_to_E": viz.EdgeLayout(label="External drive"),
            "E_to_I": viz.EdgeLayout(label=r"$J_{E\to I}$"),
            "I_to_E": viz.EdgeLayout(label=r"$\frac{g_I}{\tau_I}$"),
        },
    )
    layout.write(OUTPUT_DIR / "layout.json")
    drawing = bundle.excalidraw_export(
        OUTPUT_DIR / "network.excalidraw",
        layout=layout,
    )
    bundle.visualise(
        OUTPUT_DIR / "network-static.png",
        layout=layout,
        scale=2,
    )
    viz.render_excalidraw(
        drawing,
        OUTPUT_DIR / "network.png",
        scale=2,
    )


if __name__ == "__main__":
    main()
