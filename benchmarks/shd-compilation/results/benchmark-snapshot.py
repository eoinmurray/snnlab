"""Synthetic SHD-shaped compilation profile; no dataset/accuracy claims."""

from __future__ import annotations

import cProfile
import gc
import hashlib
import json
import pstats
import time
from pathlib import Path

import torch

from snnlab import extensions as E
from snnlab import lang
from snnlab.lang import training
from snnlab.sim.execution import GraphExecutor, plan_graph


def shd_derivative(value, config):
    return (1 / 15) / (1 + (100 / 15) * value.abs()) ** 2


def graph(
    kind,
    *,
    width=128,
    channels=700,
    inhibitory=32,
    diagnostics=False,
    drive_weight=0.03,
):
    net = lang.Network(f"shd_{kind}", dt=1 * lang.ms)
    source = net.input(
        "events", shape=("time", "batch", channels), signal_type="spikes", unit="spike"
    )
    frozen = []
    for index in range(2):
        cells = net.population(
            f"layer{index}_E",
            size=width,
            neuron=lang.COBA_LIF(
                tau_mem=20 * lang.ms, refractory=3 * lang.ms, voltage_grad_dampen=1.0
            ),
        )
        net.connect(
            source,
            cells.excitatory,
            name=f"feed{index}",
            synapse=lang.AMPA(tau=2 * lang.ms),
            weight=lang.Normal(drive_weight, drive_weight / 6),
            constraint=lang.NonNegative(),
        )
        if kind == "ping":
            interneurons = net.population(
                f"layer{index}_I",
                size=inhibitory,
                neuron=lang.COBA_LIF(
                    tau_mem=5 * lang.ms, refractory=2 * lang.ms, voltage_grad_dampen=1.0
                ),
            )
            for pre, post, polarity, synapse, weight, name in (
                (
                    cells,
                    interneurons,
                    "excitatory",
                    lang.AMPA(tau=2 * lang.ms),
                    0.1,
                    "ei",
                ),
                (
                    interneurons,
                    cells,
                    "inhibitory",
                    lang.GABA(tau=9 * lang.ms),
                    0.2,
                    "ie",
                ),
            ):
                projection = net.connect(
                    pre.spikes,
                    getattr(post, polarity),
                    name=f"loop{index}_{name}",
                    connection="recurrent",
                    delay=1 * lang.ms,
                    synapse=synapse,
                    weight=lang.Constant(weight),
                    constraint=lang.NonNegative(),
                )
                frozen.append(projection.weight.id)
            if diagnostics:
                net.output(f"i{index}", interneurons.spikes)
        if diagnostics:
            net.output(f"e{index}", cells.spikes)
            net.output(f"v{index}", cells.voltage)
        source = cells.spikes
    readout = net.population(
        "readout", size=20, neuron=lang.LeakyIntegrator(tau=20 * lang.ms), spiking=False
    )
    net.connect(
        source,
        readout.excitatory,
        name="readout_drive",
        synapse=lang.LeakyIntegrator(tau=20 * lang.ms),
        weight=lang.Normal(0.1, 0.03),
    )
    net.output("scores", readout.voltage)
    compiled = lang.compile(net).graph
    return compiled, tuple(
        row["id"] for row in net.parameters if row["id"] not in frozen
    )


def model_for(graph_data, parameters, mode, device):
    surrogate = (
        training.CustomSurrogate("benchmark.shd/v1")
        if mode == "custom"
        else training.FastSigmoid(
            slope=100 / 15, gradient_scale=0.01, threshold_inclusive=False
        )
    )
    return GraphExecutor(
        plan_graph(graph_data),
        seed=42,
        trainable_parameters=parameters,
        surrogate=surrogate.json(),
        compile_mode="default" if mode == "compiled" else None,
    ).to(device)


def forward(model, values, *, snapshots=False):
    return model(
        {"events": values},
        diagnostics=False,
        retain_parameters=snapshots,
        retain_final_state=snapshots,
        retain_runtime_state=snapshots,
    )


def run(kind, output, *, batches=(32, 256, 8), steps=1000, repeats=3, device="cuda"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        definition = E.get("surrogate", "benchmark.shd/v1")
    except ValueError:
        E.register_surrogate("benchmark.shd/v1", shd_derivative)
    else:
        if definition.function is not shd_derivative:
            raise ValueError("benchmark surrogate registration differs from reference")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    graph_data, parameters = graph(kind)
    report = {
        "kind": kind,
        "steps": steps,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name() if device == "cuda" else device,
        "input": "synthetic Bernoulli p=0.01, 700 channels, seed 43",
        "graph": graph_data,
        "rows": [],
        "source_sha256": {
            str(p.relative_to(Path(__file__).resolve().parents[2])): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in sorted((Path(__file__).resolve().parents[2] / "src").rglob("*.py"))
        },
    }
    synchronize = torch.cuda.synchronize if device == "cuda" else lambda: None

    def timed(fn):
        synchronize()
        start = time.perf_counter()
        value = fn()
        synchronize()
        return value, time.perf_counter() - start

    for batch in batches:
        generator = torch.Generator(device=device).manual_seed(43)
        values = (
            torch.rand(steps, batch, 700, device=device, generator=generator) < 0.01
        ).float()
        targets = torch.arange(batch, device=device) % 20
        for mode in ("custom", "eager", "compiled"):
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            model = model_for(graph_data, parameters, mode, device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            def update(model=model, optimizer=optimizer, values=values):
                optimizer.zero_grad(set_to_none=True)
                result = forward(model, values)
                loss = torch.nn.functional.cross_entropy(
                    result.outputs["scores"].max(dim=0).values, targets
                )
                loss.backward()
                optimizer.step()
                model.enforce_constraints()

            _, cold = timed(update)
            _, second = timed(update)
            fw, bw, updates = [], [], []
            for _ in range(repeats):
                optimizer.zero_grad(set_to_none=True)
                result, duration = timed(lambda model=model: forward(model, values))
                fw.append(duration)
                loss = torch.nn.functional.cross_entropy(
                    result.outputs["scores"].max(dim=0).values, targets
                )
                _, duration = timed(loss.backward)
                bw.append(duration)
                optimizer.step()
                model.enforce_constraints()
                _, duration = timed(update)
                updates.append(duration)
            row = {
                "batch": batch,
                "mode": mode,
                "cold_update_s": cold,
                "second_update_s": second,
                "forward_s": fw,
                "backward_s": bw,
                "update_s": updates,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated()
                if device == "cuda"
                else None,
            }
            # Profiles are separate from timing samples; costs overlap and must not be added.
            if batch == batches[0]:
                activities = [torch.profiler.ProfilerActivity.CPU]
                if device == "cuda":
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                with torch.profiler.profile(activities=activities) as profile:
                    update(values=values[:128])
                    synchronize()
                events = profile.key_averages()
                row["operators"] = [
                    {
                        "name": e.key,
                        "calls": e.count,
                        "self_cpu_us": e.self_cpu_time_total,
                        "self_device_us": e.self_device_time_total,
                    }
                    for e in events
                ]
                profile.export_chrome_trace(str(output / f"{kind}-{mode}-trace.json"))
                python_profile = cProfile.Profile()
                row["profile_steps"] = min(128, steps)
                python_profile.enable()
                update(values=values[:128])
                synchronize()
                python_profile.disable()
                python_profile.dump_stats(str(output / f"{kind}-{mode}.pstats"))
                with (output / f"{kind}-{mode}-python.txt").open("w") as handle:
                    pstats.Stats(python_profile, stream=handle).sort_stats(
                        "tottime"
                    ).print_stats(45)
                del profile, events, python_profile
            report["rows"].append(row)
            (output / f"{kind}.json").write_text(json.dumps(report, indent=2))
            print(
                json.dumps(
                    {key: value for key, value in row.items() if key != "operators"}
                ),
                flush=True,
            )
            del result, loss, update, optimizer, model
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("coba", "ping"), default="coba")
    parser.add_argument("--output", default="results")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batches", type=int, nargs="+", default=[32, 256, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(
        args.kind,
        args.output,
        batches=args.batches,
        steps=args.steps,
        repeats=args.repeats,
        device=args.device,
    )


def validate(kind, device="cuda"):
    graph_data, parameters = graph(
        kind, width=12, inhibitory=3, channels=30, diagnostics=True, drive_weight=0.3
    )
    # Strong deterministic input exercises spiking, refractory resets and delayed loops.
    values = torch.ones(33, 3, 30, device=device)
    results = []
    for mode in ("custom", "eager", "compiled"):
        model = model_for(graph_data, parameters, mode, device)
        result = forward(model, values, snapshots=True)
        loss = result.outputs["scores"].max(dim=0).values.square().mean()
        loss.backward()
        gradients = {
            name: value.grad.detach().clone()
            for name, value in model.parameter_map().items()
            if value.requires_grad
        }
        assert sum(value.abs().sum().item() for value in gradients.values()) > 0
        assert result.outputs["e0"].sum() > 0
        assert result.outputs["e1"].sum() > 0
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.step()
        model.enforce_constraints()
        weights = {
            name: value.detach().clone()
            for name, value in model.parameter_map().items()
        }
        with torch.no_grad():
            whole = forward(model, values, snapshots=True)
            head = forward(model, values[:13], snapshots=True)
            tail = model(
                {"events": values[13:]},
                diagnostics=False,
                runtime_state=head.runtime_state,
            )
            for name in whole.outputs:
                torch.testing.assert_close(
                    torch.cat((head.outputs[name], tail.outputs[name])),
                    whole.outputs[name],
                    atol=2e-5,
                    rtol=2e-5,
                )
        results.append((result.outputs, gradients, weights, optimizer.state_dict()))
    for actual in results[1:]:
        torch.testing.assert_close(actual, results[0], atol=2e-5, rtol=2e-5)
    return {
        "passed": True,
        "atol": 2e-5,
        "rtol": 2e-5,
        "steps": 33,
        "split_at": 13,
        "comparisons": [
            "custom/eager/compiled scores, voltages and spikes",
            "gradients",
            "Adam parameters and moments",
            "delayed PING, refractory and runtime continuation",
        ],
    }
