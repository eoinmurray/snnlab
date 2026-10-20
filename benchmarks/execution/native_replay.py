"""Matched public-trainer CUDA replay timing, including capture and reporting."""

import argparse
import json
import time
from pathlib import Path

import torch
from benchmark import source_provenance

from snnlab import lang, sim


def benchmark(epochs):
    net = lang.Network("native-replay-benchmark", dt=1 * lang.ms)
    source = net.input(
        "events", shape=("time", "batch", 64), signal_type="spikes", unit="spike"
    )
    cells = net.population("cells", size=32, neuron=lang.CUBA_LIF())
    for name, signal, delay in (("drive", source, 0), ("recurrent", cells.spikes, 2)):
        net.connect(
            signal,
            cells.excitatory,
            name=name,
            synapse=lang.ExponentialCurrent(tau=3 * lang.ms),
            weight=lang.Constant(4 if name == "drive" else 0.1),
            delay=delay * lang.ms,
            allow_autapses=False,
            constraint=lang.NonNegative(),
        )
    scores = lang.MeanVoltage(source=cells.spikes, classes=2, name="readout")
    net.output("scores", scores)
    recipe = lang.TrainSpec(
        objectives=(lang.CrossEntropy(prediction=scores, target="label"),),
        parameter_groups=(
            lang.ParameterGroup(
                [row["id"] for row in net.parameters], name="all", lr=0.001
            ),
        ),
        optimizer=lang.AdamW(weight_decay=0.01),
        presentation_duration=128 * lang.ms,
    )
    bundle = lang.compile(net, training=recipe)
    inputs = (
        torch.rand(128, 65, 64, generator=torch.Generator().manual_seed(17)) < 0.2
    ).float()
    kwargs = dict(
        graph=bundle.graph,
        training=bundle.training,
        input_bindings=(sim.DenseArrayBinding("events", inputs),),
        targets={"label": torch.arange(65) % 2},
        device="cuda",
        epochs=epochs,
        batch_size=32,
        seed=7,
        diagnostics=False,
        compile_block_steps=16,
        training_timing=True,
        precision="highest",
        fused_optimizer="off",
    )
    measurements = []
    results = {}
    # First pair records compiler startup. The second pair can reuse compiler
    # caches, but each train call must build its own capture cache.
    for phase in ("first", "compiler_warm"):
        for policy in ("off", "auto"):
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = sim.train(**kwargs, cuda_graphs=policy)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            measurements.append(
                dict(
                    phase=phase,
                    policy=policy,
                    total_call_s=elapsed,
                    samples_per_s=65 * epochs / elapsed,
                    training_timing=result.metrics["training_timing"],
                    cuda_graphs=result.metrics["cuda_graphs"],
                )
            )
            results[policy] = result
        for group in ("parameters", "gradients", "outputs"):
            for name, value in getattr(results["off"], group).items():
                torch.testing.assert_close(
                    getattr(results["auto"], group)[name], value, atol=3e-4, rtol=3e-4
                )
        for name, state in results["off"].optimizer_state.items():
            for key, value in state.items():
                torch.testing.assert_close(
                    results["auto"].optimizer_state[name][key],
                    value,
                    atol=3e-4,
                    rtol=3e-4,
                )
        assert results["auto"].metrics["cuda_graphs"]["replayed_updates"] == 3 * epochs
    return dict(
        schema="snnlab.native-replay-benchmark/v1",
        status="complete",
        hardware=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        provenance=source_provenance(),
        shape=[128, 65, 64],
        batch_size=32,
        epochs=epochs,
        measurements=measurements,
        checks=["outputs", "gradients", "parameters", "AdamW state"],
        limits=[
            "Synthetic workload; no accuracy claim.",
            "Complete train calls include preparation, compiler/capture startup, online metrics and checkpoint construction.",
            "Compiler-warm calls reuse compilation, but recreate captures. First auto call follows off's compiler startup; not matched clean-cache timing.",
            "Scope captures forward/loss/backward; optimizer and constraints retain normal execution.",
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    if args.epochs < 2:
        parser.error("epochs must be at least two")
    report = benchmark(args.epochs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["measurements"], indent=2))
