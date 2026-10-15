"""Run step, ramp, sine and custom schedules, including portable continuation.

From the repository root:
    python examples/weight-modulation/weight_modulation.py
"""

import argparse
from pathlib import Path

import torch

from snnlab import lang
from snnlab.sim import GraphExecutor, plan_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/weight-modulation")
    )
    args = parser.parse_args()
    net = lang.Network("weight_modulation", dt=1 * lang.ms)
    inputs = net.input(
        "events", shape=("time", "batch", 1), signal_type="spikes", unit="spike"
    )
    schedules = {
        "step": lang.StepGain(at=50 * lang.ms),
        "ramp": lang.RampGain(start=25 * lang.ms, end=75 * lang.ms),
        "sine": lang.SineGain(frequency=40 * lang.Hz),
        "blocks": lang.PiecewiseGain(
            times=[25 * lang.ms, 75 * lang.ms], values=[1, 0, 2]
        ),
        "custom": lang.CustomGain(
            "timing_functions:gated_sine",
            config={"start_ms": 25, "frequency_hz": 40, "depth": 0.5},
        ),
    }
    for name, schedule in schedules.items():
        cells = net.population(
            f"{name}_cells",
            size=1,
            neuron=lang.CUBA_LIF(threshold_mv=1000),
        )
        projection = net.connect(
            inputs,
            cells.excitatory,
            name=f"{name}_input",
            synapse=lang.ExponentialCurrent(tau=5 * lang.ms),
            weight=lang.Constant(1),
            modulation=schedule,
        )
        net.output(f"{name}_current", projection.current)

    path = lang.compile(net, target="tools/snnsim").write(
        args.output / "network.bundle"
    )
    bundle = lang.load_bundle(path)
    model = GraphExecutor(plan_graph(bundle.graph))
    events = torch.ones(100, 1, 1)
    full = model({"events": events})
    first = model({"events": events[:43]})
    second = model({"events": events[43:]}, runtime_state=first.runtime_state)
    for name, values in full.outputs.items():
        torch.testing.assert_close(
            values,
            torch.cat([first.outputs[name], second.outputs[name]]),
            rtol=0,
            atol=0,
        )
        print(f"{name}: t=0 {values[0].item():.4f} nA; t=99 {values[-1].item():.4f} nA")
    print(f"Bundle: {path.resolve()}; split-run equality verified")


if __name__ == "__main__":
    main()
