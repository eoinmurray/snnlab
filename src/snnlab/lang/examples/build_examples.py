"""Build all representative v1 bundles. Run from the repository root."""

from pathlib import Path

from snnlab import lang as snn
from snnlab.lang import training

OUT = Path.cwd() / "examples" / "generated"


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


def coupled_feedback():
    net = snn.Network("coupled_ping_feedback")
    stimulus = net.input(
        "stimulus", shape=("time", "batch", 128), signal_type="spikes", unit="spike"
    )
    left = snn.components.ping(net, name="object_a", n_e=192, n_i=48, source=stimulus)
    right = snn.components.ping(net, name="object_b", n_e=192, n_i=48, source=stimulus)
    net.connect(
        left.E.spikes,
        right.E.excitatory,
        name="a_to_b",
        synapse=snn.AMPA(tau=5 * snn.ms),
        weight=snn.Normal(0.15, 0.02),
        connection="feedforward",
        delay=1 * snn.ms,
    )
    net.connect(
        right.E.spikes,
        left.E.modulatory,
        name="b_feedback_a",
        synapse=snn.Modulatory(tau=12 * snn.ms),
        weight=snn.Normal(0.1, 0.01),
        connection="feedback",
        delay=1 * snn.ms,
    )
    net.expose(left.E.spikes, right.E.spikes, name="coupled")
    return snn.compile(net)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, bundle in {
        "ping_classifier": ping_classifier(),
        "deep_network": deep_network(),
        "coupled_feedback": coupled_feedback(),
    }.items():
        bundle.write(OUT / f"{name}.bundle", visualise=True)
        print(OUT / f"{name}.bundle")


if __name__ == "__main__":
    main()
