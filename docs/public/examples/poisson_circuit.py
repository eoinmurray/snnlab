"""Run a small excitatory/inhibitory circuit with seeded Poisson drive."""

from pathlib import Path

import numpy as np
import torch

from snnlab import lang as snn
from snnlab.sim.execution import ExecutionSpec, PoissonInputBinding, simulate


def main(out=Path("artifacts/examples/poisson_circuit")):
    net = snn.Network("driven_ei", dt=0.1 * snn.ms)
    events = net.input(
        "events", shape=("time", "batch", 4), signal_type="spikes", unit="spike"
    )
    cell = snn.components.ping(
        net, name="cell", n_e=8, n_i=2, source=events, w_in=snn.Constant(0.5)
    )
    net.output("excitatory_spikes", cell.E.spikes)
    net.output("inhibitory_spikes", cell.I.spikes)
    bundle = snn.compile(net, target="tools/snnsim")
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=bundle.graph,
            poisson_bindings=(
                PoissonInputBinding(
                    input_id="events",
                    steps_count=1000,
                    batch_size=1,
                    rates_hz=(100.0,),
                    seed=17,
                ),
            ),
            seed=17,
            device="cpu",
        )
    )
    duration_seconds = 1000 * 0.1 / 1000
    for name, spikes in result.outputs.items():
        assert torch.isfinite(spikes).all()
        rate_hz = spikes.sum().item() / (
            spikes.shape[1] * spikes.shape[2] * duration_seconds
        )
        print(f"{name}: shape={tuple(spikes.shape)}, mean rate={rate_hz:.1f} Hz")
    Path(out).mkdir(parents=True, exist_ok=True)
    np.savez(
        Path(out) / "activity.npz",
        dt_ms=0.1,
        **{name: value.detach().numpy() for name, value in result.outputs.items()},
    )


if __name__ == "__main__":
    main()
