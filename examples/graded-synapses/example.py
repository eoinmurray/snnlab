import shutil
from pathlib import Path

import numpy as np
import torch

from snnlab import extensions as ext
from snnlab import lang
from snnlab.sim import GraphExecutor, plan_graph

OUTPUT_DIR = Path(__file__).resolve().parent / "results"

DT_MS = 0.1


def prepare_results():
    if OUTPUT_DIR.is_symlink():
        OUTPUT_DIR.unlink()
    elif OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)


def membrane_init(ctx):
    voltage = torch.full(
        ctx.shape, ctx.config["initial_mv"], device=ctx.device, dtype=ctx.dtype
    )
    return {"voltage": voltage, "activation": voltage.clone()}


def membrane_step(ctx):
    voltage = ctx.state["voltage"]
    current = (
        0.01 * (-60.0 - voltage)
        + ctx.config["bias_na"]
        + ctx.excitatory * (0.0 - voltage)
        + ctx.inhibitory * (-80.0 - voltage)
    )
    updated = voltage + ctx.dt_ms * current / 0.2
    return {
        **ctx.state,
        "voltage": updated,
        "activation": updated + 5.0,
    }, torch.zeros_like(voltage)


def graded_init(ctx):
    gate = torch.full(ctx.shape, -60.0, device=ctx.device, dtype=ctx.dtype)
    return {"gate": gate, "value": torch.zeros_like(gate)}


def graded_step(ctx):
    gate = ctx.state["gate"] + ctx.dt_ms / 2.0 * (ctx.drive - ctx.state["gate"])
    conductance = 0.04 * torch.sigmoid(0.3 * (gate + 50.0))
    return {"gate": gate, "value": conductance}


def register():
    names = {definition.name for definition in ext.definitions()}
    if "example.graded_membrane/v1" not in names:
        ext.register_neuron(
            "example.graded_membrane/v1",
            membrane_step,
            initialize=membrane_init,
            input_unit="uS",
            state_units={"activation": "mV"},
        )
    if "example.graded_synapse/v1" not in names:
        ext.register_synapse(
            "example.graded_synapse/v1",
            graded_step,
            initialize=graded_init,
            input_unit="mV",
            output_unit="uS",
            state_units={"gate": "mV"},
        )


def network(*, delay_steps=0, state_source=False, weight=1.0):
    register()
    net = lang.Network("graded", dt=DT_MS * lang.ms)

    net.input("clock", shape=("time", "batch", 1), signal_type="continuous", unit="1")
    a = net.population(
        "a",
        size=1,
        spiking=False,
        neuron=lang.CustomNeuron(
            "example.graded_membrane/v1", initial_mv=-40.0, bias_na=0.2
        ),
    )
    b = net.population(
        "b",
        size=1,
        spiking=False,
        neuron=lang.CustomNeuron(
            "example.graded_membrane/v1", initial_mv=-65.0, bias_na=0.1
        ),
    )
    for pre, post in ((a, b), (b, a)):
        edge = net.connect(
            pre.state("activation") if state_source else pre.voltage,
            post.inhibitory,
            name=f"{pre.id}_to_{post.id}",
            synapse=lang.CustomSynapse("example.graded_synapse/v1"),
            weight=lang.Constant(weight),
            initialization_scaling="direct",
            delay=delay_steps * DT_MS * lang.ms,
            connection="recurrent",
        )
        net.output(f"{edge.id}_gate", edge.state("gate"))
        net.output(f"{edge.id}_conductance", edge.conductance)
    net.output("a_voltage", a.voltage)
    net.output("b_voltage", b.voltage)
    return net


def run(steps=200, batch=1):
    bundle = lang.compile(network())
    model = GraphExecutor(plan_graph(bundle.graph), compile_mode=None).double()
    return model({"clock": torch.zeros(steps, batch, 1, dtype=torch.float64)})


if __name__ == "__main__":
    prepare_results()

    from example import network, run

    from snnlab import viz

    diagram_network = network()

    diagram_network.outputs = [
        row for row in diagram_network.outputs if row["signal"].endswith(".voltage")
    ]
    viz.render_diagram(
        lang.diagram(lang.compile(diagram_network), view="expanded"),
        OUTPUT_DIR / "network.png",
    )

    result = run()
    keys = (
        "a_to_b_gate",
        "b_to_a_gate",
        "a_to_b_conductance",
        "b_to_a_conductance",
        "a_voltage",
        "b_voltage",
    )
    actual = np.column_stack(
        [result.outputs[key][:, 0, 0].detach().numpy() for key in keys]
    )
    np.savetxt(
        OUTPUT_DIR / "trajectory.csv",
        actual,
        delimiter=",",
        header=",".join(keys),
    )
