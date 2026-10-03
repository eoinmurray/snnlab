"""Author a feedforward circuit and round-trip its portable bundle."""

from pathlib import Path

from snnlab import lang as snn
from snnlab.lang.compiler import load_bundle
from snnlab.sim.execution import plan_graph


def main(out=Path("artifacts/examples/build_bundle")):
    net = snn.Network("feedforward", dt=0.1 * snn.ms)
    events = net.input(
        "events", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
    )
    cells = net.population("cells", size=4, neuron=snn.COBA_LIF(tau_mem=20 * snn.ms))
    net.connect(
        events,
        cells.excitatory,
        name="input_to_cells",
        synapse=snn.AMPA(tau=2 * snn.ms),
        weight=snn.Constant(0.5),
        constraint=snn.NonNegative(),
        connection="feedforward",
    )
    net.output("spikes", cells.spikes)
    net.expose(cells.voltage, name="voltage")

    bundle = snn.compile(net, target="tools/snnsim")
    path = bundle.write(Path(out) / "feedforward.bundle")
    restored = load_bundle(path)
    assert restored.graph == bundle.graph
    plan = plan_graph(restored.graph)
    assert len(plan.populations) == 1
    print("Bundle round-trip: identical graph")
    print(
        f"Plan: {len(plan.populations)} population, {len(plan.projections)} projection"
    )


if __name__ == "__main__":
    main()
