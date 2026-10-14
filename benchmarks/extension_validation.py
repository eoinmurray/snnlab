"""Synthetic custom-neuron validation benchmark (not full MNIST training).

Run from the checkout: uv run python benchmarks/extension_validation.py --device cuda
Timings and operator profiles are collected separately. Initialization remains
inside forward; explicit timing-boundary synchronization is retained.
"""

import argparse
import json
import random
import statistics
import time

import torch

from snnlab import extensions as E
from snnlab import lang
from snnlab.lang import training as recipe
from snnlab.sim.execution import GraphExecutor, plan_graph


def initialize(ctx):
    return {
        "voltage": torch.zeros(ctx.shape, device=ctx.device, dtype=ctx.dtype),
        "synaptic": torch.zeros(ctx.shape, device=ctx.device, dtype=ctx.dtype),
        "refractory": torch.zeros(ctx.shape, device=ctx.device, dtype=torch.long),
    }


def step(ctx):
    previous = ctx.state
    synaptic = 0.8 * previous["synaptic"] + ctx.excitatory - ctx.inhibitory
    reset = (previous["voltage"] > 1).to(synaptic.dtype).detach()
    voltage = 0.9 * previous["voltage"] + synaptic - reset
    return {**previous, "voltage": voltage, "synaptic": synaptic}, ctx.spike(
        voltage - 1
    )


def derivative(x, config):
    return 1 / (1 + 25 * x.abs()) ** 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repetitions", type=int, default=15)
    parser.add_argument("--warmups", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 1 or args.warmups < 0:
        parser.error("repetitions must be positive and warmups nonnegative")
    device = torch.device(args.device)
    torch.set_num_threads(4)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    E.register_neuron("benchmark.current/v1", step, initialize=initialize)
    E.register_surrogate("benchmark.sigmoid/v1", derivative)
    net = lang.Network("validation_benchmark", dt=1 * lang.ms)
    source = net.input(
        "inputs", shape=("time", "batch", 784), signal_type="spikes", unit="spike"
    )
    for name, size in (("hidden", 256), ("output", 10)):
        cells = net.population(
            name, size=size, neuron=lang.CustomNeuron("benchmark.current/v1")
        )
        net.connect(
            source,
            cells.excitatory,
            name=f"to_{name}",
            synapse=lang.ExponentialCurrent(tau=3 * lang.ms),
            weight=lang.Normal(mean=0, std=0.2),
        )
        source = cells.spikes
    net.output("spikes", source)
    plan = plan_graph(lang.compile(net).graph)
    models = {
        policy: GraphExecutor(
            plan,
            seed=7,
            trainable_parameters=tuple(p.parameter for p in plan.projections),
            surrogate=recipe.CustomSurrogate("benchmark.sigmoid/v1").json(),
            extension_validation=policy,
        ).to(device)
        for policy in ("strict", "structural")
    }
    torch.manual_seed(8)
    inputs = (torch.rand(50, 128, 784, device=device) < 0.1).float()
    labels = torch.arange(128, device=device) % 10

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    def operation(model, backward):
        with torch.set_grad_enabled(backward):
            result = model({"inputs": inputs}, diagnostics=False)
            scores = result.outputs["spikes"].sum(0)
            if backward:
                torch.nn.functional.cross_entropy(scores, labels).backward()
        return scores

    reference = None
    for model in models.values():
        model.zero_grad(set_to_none=True)
        scores = operation(model, True)
        values = [scores.detach()] + [
            p.grad.detach().clone() for p in model.parameters()
        ]
        if reference is None:
            reference = values
        else:
            for actual, expected in zip(values, reference):
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    jobs = [(policy, backward) for policy in models for backward in (False, True)]
    for _ in range(args.warmups):
        for policy, backward in jobs:
            models[policy].zero_grad(set_to_none=True)
            operation(models[policy], backward)
    timings = {job: [] for job in jobs}
    rng = random.Random(7)
    for _ in range(args.repetitions):
        rng.shuffle(jobs)
        for policy, backward in jobs:
            model = models[policy]
            model.zero_grad(set_to_none=True)
            synchronize()
            start = time.perf_counter()
            operation(model, backward)
            synchronize()
            timings[policy, backward].append((time.perf_counter() - start) * 1000)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    rows = []
    for policy, backward in jobs:
        models[policy].zero_grad(set_to_none=True)
        synchronize()
        with torch.profiler.profile(activities=activities) as profile:
            operation(models[policy], backward)
            synchronize()
        keys = {"aten::isfinite", "aten::_local_scalar_dense", "cudaStreamSynchronize"}
        rows.append(
            {
                "policy": policy,
                "backward": backward,
                "median_ms": statistics.median(timings[policy, backward]),
                "operator_counts": {
                    row.key: row.count
                    for row in profile.key_averages()
                    if row.key in keys
                },
            }
        )
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None,
                "warmups": args.warmups,
                "repetitions": args.repetitions,
                "score_gradient_agreement": True,
                "results": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
