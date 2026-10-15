"""Define two layers with local inhibitory feedback and draw them. Requires Graphviz."""

from pathlib import Path

from snnlab import lang, viz

net = lang.Network("two_layer", dt=0.1 * lang.ms)
inputs = net.input(
    "inputs", shape=("time", "batch", 32), signal_type="spikes", unit="spike"
)

layers = []
for name, e_size, i_size in [("layer_1", 64, 16), ("layer_2", 32, 8)]:
    with net.group(name):
        e = net.population(
            f"{name}_E", size=e_size,
            neuron=lang.COBA_LIF(tau_mem=20 * lang.ms),
        )
        i = net.population(
            f"{name}_I", size=i_size,
            neuron=lang.COBA_LIF(tau_mem=5 * lang.ms),
        )
        net.connect(
            e.spikes, i.excitatory, name=f"{name}_E_to_I",
            synapse=lang.AMPA(tau=2 * lang.ms),
            weight=lang.Uniform(0.1, 0.5), constraint=lang.NonNegative(),
            connection="recurrent", delay=0.1 * lang.ms,
        )
        net.connect(
            i.spikes, e.inhibitory, name=f"{name}_I_to_E",
            synapse=lang.GABA(tau=9 * lang.ms),
            weight=lang.Uniform(0.5, 1.0), constraint=lang.NonNegative(),
            connection="feedback", delay=0.1 * lang.ms,
        )
    layers.append(e)

net.connect(
    inputs, layers[0].excitatory, name="input_to_layer_1",
    synapse=lang.AMPA(tau=2 * lang.ms),
    weight=lang.Uniform(0.1, 0.5), constraint=lang.NonNegative(),
)
net.connect(
    layers[0].spikes, layers[1].excitatory, name="layer_1_to_layer_2",
    synapse=lang.AMPA(tau=2 * lang.ms),
    weight=lang.Uniform(0.1, 0.5), constraint=lang.NonNegative(),
)
net.output("spikes", layers[1].spikes)

bundle = lang.compile(net)
viz.render_diagram(
    lang.diagram(bundle, view="expanded"),
    Path(__file__).with_suffix(".png"), scale=2,
)
