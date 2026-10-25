import json
import shutil
from pathlib import Path

import numpy as np
import torch

from snnlab import lang, viz
from snnlab.sim import (
    DenseArrayBinding,
    simulate,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "results"

DT_MS = 0.25
DURATION_MS = 360
CURRENT_NA = 5.6
TAU_MS = 30
CAPACITANCE_NF = 0.2


def prepare_results():
    if OUTPUT_DIR.is_symlink():
        OUTPUT_DIR.unlink()
    elif OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)


def record_neuron():
    net = lang.Network(
        "animated_neuron",
        dt=DT_MS * lang.ms,
    )
    inputs = net.input(
        "pulses",
        shape=("time", "batch", 1),
        signal_type="spikes",
        unit="spike",
    )
    cell = net.population(
        "neuron",
        size=1,
        neuron=lang.CUBA_LIF(
            tau_mem=TAU_MS * lang.ms,
            capacitance_nf=CAPACITANCE_NF,
        ),
    )

    projection = net.connect(
        inputs,
        cell.excitatory,
        name="drive",
        synapse=lang.ExponentialCurrent(tau=0.001 * lang.ms),
        weight=lang.Constant(CURRENT_NA),
        initialization_scaling="direct",
    )
    net.output(
        "spikes",
        cell.spikes,
    )
    net.expose(
        cell.voltage,
        name="voltage",
    )
    net.expose(
        projection.current,
        name="current",
    )
    bundle = lang.compile(net)

    departures = np.arange(
        8.0,
        DURATION_MS - 6,
        12.0,
    )
    arrivals = departures + 6
    pulses = torch.zeros(
        round(DURATION_MS / DT_MS),
        1,
        1,
    )
    pulses[np.rint(arrivals / DT_MS).astype(int), 0, 0] = 1
    result = simulate(
        graph=bundle.graph,
        device="cpu",
        seed=17,
        input_bindings=(
            DenseArrayBinding(
                "pulses",
                pulses,
            ),
        ),
    ).numpy(batch=0)

    time_ms = np.arange(pulses.shape[0] + 1) * DT_MS
    voltage = np.r_[-65.0, result.diagnostics["voltage"][:, 0]]
    spike_steps = np.flatnonzero(result.outputs["spikes"][:, 0])
    recording = {
        "schema": "snnlab.neuron-replay/v1",
        "dt_ms": DT_MS,
        "duration_ms": DURATION_MS,
        "rest_mv": -65,
        "threshold_mv": -50,
        "capacitance_nf": CAPACITANCE_NF,
        "tau_ms": TAU_MS,
        "current_na": CURRENT_NA,
        "time_convention": "Input at slot start; voltage and spikes at integration end.",
        "input_departures_ms": departures.tolist(),
        "input_arrivals_ms": arrivals.tolist(),
        "time_ms": time_ms.tolist(),
        "voltage_mv": voltage.tolist(),
        "spike_times_ms": ((spike_steps + 1) * DT_MS).tolist(),
    }
    return bundle, recording, result


if __name__ == "__main__":
    prepare_results()
    output = OUTPUT_DIR
    bundle, recording, _ = record_neuron()
    (output / "recording.json").write_text(
        json.dumps(
            recording,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    viz.render_diagram(
        lang.diagram(bundle),
        output / "network.png",
        scale=2,
    )
