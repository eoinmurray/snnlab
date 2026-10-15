"""Define a feedforward network and render its diagram. Requires Graphviz."""

from pathlib import Path

from snnlab import lang, viz

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
    inputs, hidden.excitatory, name="input_to_hidden",
    synapse=lang.AMPA(tau=2 * lang.ms),
    weight=lang.Uniform(0.1, 0.5), constraint=lang.NonNegative(),
)
net.connect(
    hidden.spikes, output.excitatory, name="hidden_to_output",
    synapse=lang.AMPA(tau=2 * lang.ms),
    weight=lang.Uniform(0.1, 0.5), constraint=lang.NonNegative(),
)
net.output("spikes", output.spikes)

bundle = lang.compile(net)
viz.render_diagram(
    lang.diagram(bundle, view="expanded"),
    Path(__file__).with_suffix(".png"), scale=2,
)
