import shutil
from pathlib import Path

from snnlab import lang, viz

OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def prepare_results():
    if OUTPUT_DIR.is_symlink():
        OUTPUT_DIR.unlink()
    elif OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)


def feedforward():
    net = lang.Network("feedforward", dt=0.1 * lang.ms)
    inputs = net.input(
        "inputs", shape=("time", "batch", 16), signal_type="spikes", unit="spike"
    )
    hidden = net.population(
        "hidden", size=32, neuron=lang.COBA_LIF(tau_mem=20 * lang.ms)
    )
    output = net.population(
        "output", size=8, neuron=lang.COBA_LIF(tau_mem=20 * lang.ms)
    )

    net.connect(
        inputs,
        hidden.excitatory,
        name="input_to_hidden",
        synapse=lang.AMPA(tau=2 * lang.ms),
        weight=lang.Uniform(0.1, 0.5),
        constraint=lang.NonNegative(),
    )
    net.connect(
        hidden.spikes,
        output.excitatory,
        name="hidden_to_output",
        synapse=lang.AMPA(tau=2 * lang.ms),
        weight=lang.Uniform(0.1, 0.5),
        constraint=lang.NonNegative(),
    )
    net.output("spikes", output.spikes)

    bundle = lang.compile(net)
    viz.render_diagram(
        lang.diagram(bundle, view="expanded"),
        OUTPUT_DIR / "feedforward.png",
        scale=2,
    )


def recurrent():
    net = lang.Network("two_layer", dt=0.1 * lang.ms)
    inputs = net.input(
        "inputs", shape=("time", "batch", 32), signal_type="spikes", unit="spike"
    )

    layers = []
    for name, e_size, i_size in [("layer_1", 64, 16), ("layer_2", 32, 8)]:
        with net.group(name):
            e = net.population(
                f"{name}_E",
                size=e_size,
                neuron=lang.COBA_LIF(tau_mem=20 * lang.ms),
            )
            i = net.population(
                f"{name}_I",
                size=i_size,
                neuron=lang.COBA_LIF(tau_mem=5 * lang.ms),
            )
            net.connect(
                e.spikes,
                i.excitatory,
                name=f"{name}_E_to_I",
                synapse=lang.AMPA(tau=2 * lang.ms),
                weight=lang.Uniform(0.1, 0.5),
                constraint=lang.NonNegative(),
                connection="recurrent",
                delay=0.1 * lang.ms,
            )
            net.connect(
                i.spikes,
                e.inhibitory,
                name=f"{name}_I_to_E",
                synapse=lang.GABA(tau=9 * lang.ms),
                weight=lang.Uniform(0.5, 1.0),
                constraint=lang.NonNegative(),
                connection="feedback",
                delay=0.1 * lang.ms,
            )
        layers.append(e)

    net.connect(
        inputs,
        layers[0].excitatory,
        name="input_to_layer_1",
        synapse=lang.AMPA(tau=2 * lang.ms),
        weight=lang.Uniform(0.1, 0.5),
        constraint=lang.NonNegative(),
    )
    net.connect(
        layers[0].spikes,
        layers[1].excitatory,
        name="layer_1_to_layer_2",
        synapse=lang.AMPA(tau=2 * lang.ms),
        weight=lang.Uniform(0.1, 0.5),
        constraint=lang.NonNegative(),
    )
    net.output("spikes", layers[1].spikes)

    bundle = lang.compile(net)
    viz.render_diagram(
        lang.diagram(bundle, view="expanded"),
        OUTPUT_DIR / "recurrent.png",
        scale=2,
    )


def main():
    prepare_results()
    feedforward()
    recurrent()


if __name__ == "__main__":
    main()
