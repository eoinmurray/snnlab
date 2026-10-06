"""Shared graph builders for focused execution test modules."""

from snnlab import lang as snn
from snnlab.lang import training
from tests._circuits import author_ping


def state_tensors(state):
    for group in (
        state.voltages,
        state.refractory,
        state.conductances,
        state.population_histories,
        state.input_histories,
    ):
        yield from group.values()


def coupled_graph(*, direction="reciprocal", delay_ms=0.1):
    net = snn.Network("coupled_ping_gate", dt=0.1 * snn.ms)
    drive_a = net.input(
        "drive_a", shape=("time", "batch", 3), signal_type="spikes", unit="spike"
    )
    drive_b = net.input(
        "drive_b", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
    )
    a = author_ping(net, name="a", n_e=4, n_i=1, source=drive_a)
    b = author_ping(net, name="b", n_e=6, n_i=2, source=drive_b)
    if direction in {"unidirectional", "reciprocal"}:
        net.connect(
            a.I.spikes,
            b.E.inhibitory,
            name="a_I_to_b_E",
            synapse=snn.GABA(tau=9 * snn.ms),
            weight=snn.Constant(4.0),
            constraint=snn.NonNegative(),
            connection="feedback",
            delay=delay_ms * snn.ms,
        )
    if direction == "reciprocal":
        net.connect(
            b.I.spikes,
            a.E.inhibitory,
            name="b_I_to_a_E",
            synapse=snn.GABA(tau=9 * snn.ms),
            weight=snn.Constant(4.0),
            constraint=snn.NonNegative(),
            connection="feedback",
            delay=delay_ms * snn.ms,
        )
    net.expose(a.E.spikes, a.I.spikes, b.E.spikes, b.I.spikes, name="coupled")
    net.output("a_spikes", a.E.spikes)
    return expose_graph_diagnostics(snn.compile(net, target=None).graph)


def direct_train_bundle():
    net = snn.Network("direct_train", dt=0.1 * snn.ms)
    events = net.input(
        "events", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
    )
    scores = snn.readouts.SpikeCount(source=events, classes=2, name="scores")
    snn.ops.linear(events, size=1, name="shadow")
    net.output("class_scores", scores)
    parameter_ids = [row["id"] for row in net.parameters]
    recipe = snn.TrainSpec(
        objectives=[training.CrossEntropy(prediction=scores, target="label")],
        parameter_groups=[
            training.ParameterGroup(
                ["scores_projection.weight"], name="readout", lr=0.1
            ),
            training.ParameterGroup(
                [pid for pid in parameter_ids if pid != "scores_projection.weight"],
                name="frozen",
                lr=0,
                frozen=True,
            ),
        ],
        optimizer=training.AdamW(weight_decay=0.0),
        presentation_duration=0.3 * snn.ms,
    )
    return snn.compile(net, training=recipe, target=None)


def standard_readout_graph(
    readout: str, *, duration: float | None = None, mask: bool = False
):
    net = snn.Network("readout_fixture", dt=100 * snn.ms)
    source = net.input(
        "events", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
    )
    valid = (
        net.input("valid", shape=("time", "batch"), signal_type="mask")
        if mask
        else None
    )
    if readout == "final":
        value = snn.readouts.FinalVoltage(source=source, classes=2, name="scores")
    elif readout == "count":
        value = snn.readouts.SpikeCount(source=source, classes=2, name="scores")
    elif readout == "rate":
        value = snn.readouts.SpikeRate(
            source=source,
            classes=2,
            name="scores",
            duration=duration,
            mask=valid,
        )
    elif readout == "cumulative":
        value = snn.readouts.CumulativePotential(
            source=source, classes=2, name="scores"
        )
    else:
        raise ValueError(readout)
    net.output("class_scores", value)
    graph = snn.compile(net, target="tools/snnsim").graph
    for parameter in graph["parameters"]:
        parameter["initializer"] = {"kind": "constant", "value": 1.0}
    return graph


def expose_graph_diagnostics(graph):
    """Explicitly declare internal traces used by numerical regression fixtures."""
    rows = graph.setdefault("observables", [])
    names = {row["id"] for row in rows}
    signals = [
        f"{row['id']}.{port}"
        for row in graph["populations"]
        for port in (["voltage", "spikes"] if row["spiking"] else ["voltage"])
    ] + [f"{row['id']}.conductance" for row in graph["projections"]]
    rows.extend(
        {"id": signal, "signal": signal} for signal in signals if signal not in names
    )
    return graph
