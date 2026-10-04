"""Shared classifier bundles for simulator regression tests."""

from snnlab import lang as snn
from snnlab.lang import training


def ping_classifier():
    net = snn.Network("ping_classifier")
    image = net.input(
        "image", shape=("time", "batch", 784), signal_type="spikes", unit="spike"
    )
    cell = snn.components.ping(
        net,
        name="sensory_ping",
        n_e=256,
        n_i=64,
        source=image,
        include_silent_recurrence=True,
    )
    readout = snn.readouts.MeanVoltage(
        source=cell.E.spikes,
        classes=10,
        name="classifier",
        tau=2 * snn.ms,
        weight=snn.Normal(5.1, 3.8),
    )
    net.output("class_logits", readout)
    net.expose(cell.E.spikes, cell.I.spikes, name="cell")
    recurrent_projection_ids = {
        projection["parameters"][0]
        for projection in net.projections
        if projection["connection"] == "recurrent"
    }
    recurrent = [p["id"] for p in net.parameters if p["id"] in recurrent_projection_ids]
    feedforward = [p["id"] for p in net.parameters if p["id"] not in set(recurrent)]
    train = snn.TrainSpec(
        objectives=[training.CrossEntropy(prediction=readout, target="digit")],
        parameter_groups=[
            training.ParameterGroup(feedforward, name="feedforward", lr=1e-3),
            training.ParameterGroup(
                recurrent,
                name="recurrent_frozen",
                lr=0.0,
                frozen=True,
            ),
        ],
        optimizer=training.AdamW(weight_decay=1e-4),
        surrogate=training.FastSigmoid(slope=1.0),
        presentation_duration=200 * snn.ms,
        epochs=20,
    )
    return snn.compile(net, training=train)


def deep_network():
    net = snn.Network("deep_ping_hierarchy")
    events = net.input(
        "events", shape=("time", "batch", 700), signal_type="spikes", unit="spike"
    )
    first = snn.components.ping(net, name="encoder", n_e=384, n_i=96, source=events)
    second = snn.components.ping(
        net, name="association", n_e=256, n_i=64, source=first.E.spikes
    )
    third = snn.components.ping(
        net, name="decision", n_e=128, n_i=32, source=second.E.spikes
    )
    result = snn.readouts.SpikeCount(
        source=third.E.spikes, classes=20, name="gesture_readout"
    )
    net.output("gesture_logits", result)
    net.expose(first.E.spikes, second.E.spikes, third.E.spikes)
    recipe = snn.TrainSpec(
        objectives=[training.CrossEntropy(prediction=result, target="gesture")],
        regularizers=[
            training.SpikeBudgetPenalty(
                signals=(
                    first.E.spikes,
                    first.I.spikes,
                    second.E.spikes,
                    second.I.spikes,
                    third.E.spikes,
                    third.I.spikes,
                ),
                ceiling_hz=100.0,
                strength=1e-4,
            )
        ],
        parameter_groups=[
            training.ParameterGroup(
                [row["id"] for row in net.parameters],
                name="all_layers",
                lr=1e-3,
            )
        ],
        optimizer=training.AdamW(weight_decay=1e-4),
        surrogate=training.FastSigmoid(slope=1.0),
        presentation_duration=100 * snn.ms,
        epochs=20,
    )
    return snn.compile(net, training=recipe)
