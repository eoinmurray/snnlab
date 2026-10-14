"""Controlled zero-decay ablation and complete matched MNIST training/test runs.

Uses the issue's 784/256/10, 50-step, alpha=0.8, beta=0.9 architecture.
Retained Pinglab checkpoints are not required: this produces new trained states.
The snnTorch reference uses explicit matrix/bias projections with identical
layouts; it is not a benchmark of fused nn.Linear or of an optimized deployment.
"""

import argparse
import inspect
import json
import math
import random
import statistics
import time
from pathlib import Path

import numpy as np
import snntorch
import torch
from torch import nn

from snnlab import extensions as E
from snnlab import lang
from snnlab.lang import training as recipe
from snnlab.sim.execution import GraphExecutor, plan_graph

NEURON = "snnlab.benchmark-synaptic/v1"
SURROGATE = "snnlab.benchmark-fast-sigmoid/v1"
NAMES = ("input_weight", "input_bias", "output_weight", "output_bias")


def initialize(ctx):
    zero = torch.zeros(ctx.shape, device=ctx.device, dtype=ctx.dtype)
    return {
        "voltage": zero,
        "synaptic": zero.clone(),
        "refractory": torch.zeros(ctx.shape, device=ctx.device, dtype=torch.long),
    }


def neuron_step(ctx):
    state = ctx.state
    reset = (state["voltage"] > 1).to(state["voltage"].dtype).detach()
    synaptic = 0.8 * state["synaptic"] + ctx.excitatory - ctx.inhibitory
    voltage = 0.9 * state["voltage"] + synaptic - reset
    return {**state, "voltage": voltage, "synaptic": synaptic}, ctx.spike(voltage - 1)


def derivative(x, config):
    return 1 / (1 + 25 * x.abs()) ** 2


def initial_weights(seed=42):
    generator = torch.Generator().manual_seed(seed)
    result = {}
    for prefix, source, target in (("input", 784, 256), ("output", 256, 10)):
        scale = 1 / math.sqrt(source)
        result[prefix + "_weight"] = torch.empty(source, target).uniform_(
            -scale, scale, generator=generator
        )
        result[prefix + "_bias"] = torch.empty(1, target).uniform_(
            -scale, scale, generator=generator
        )
    return result


class TorchModel(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.values = nn.ParameterDict(
            {name: nn.Parameter(value.clone()) for name, value in weights.items()}
        )
        self.hidden = snntorch.Synaptic(
            alpha=0.8, beta=0.9, spike_grad=snntorch.surrogate.fast_sigmoid(slope=25)
        )
        self.output = snntorch.Synaptic(
            alpha=0.8, beta=0.9, spike_grad=snntorch.surrogate.fast_sigmoid(slope=25)
        )

    def parameters_by_name(self):
        return dict(self.values.items())

    def forward(self, events):
        synaptic = torch.zeros(events.shape[1], 256, device=events.device)
        voltage = torch.zeros_like(synaptic)
        out_synaptic = torch.zeros(events.shape[1], 10, device=events.device)
        out_voltage = torch.zeros_like(out_synaptic)
        spikes = []
        for frame in events:
            hidden, synaptic, voltage = self.hidden(
                frame @ self.values["input_weight"] + self.values["input_bias"],
                synaptic,
                voltage,
            )
            output, out_synaptic, out_voltage = self.output(
                hidden @ self.values["output_weight"] + self.values["output_bias"],
                out_synaptic,
                out_voltage,
            )
            spikes.append(output)
        return torch.stack(spikes).sum(0)


class GraphModel(nn.Module):
    def __init__(self, weights, mode="fast", snapshots=True):
        super().__init__()
        try:
            E.get("neuron", NEURON)
        except ValueError:
            E.register_neuron(
                NEURON,
                neuron_step,
                initialize=initialize,
                state_units={"synaptic": "nA"},
            )
        try:
            E.get("surrogate", SURROGATE)
        except ValueError:
            E.register_surrogate(SURROGATE, derivative)
        net = lang.Network("zero_decay_matched", dt=1 * lang.ms)
        source = net.input(
            "events", shape=("time", "batch", 784), signal_type="spikes", unit="spike"
        )
        bias = net.input(
            "bias", shape=("time", "batch", 1), signal_type="spikes", unit="spike"
        )
        self.ids = {}
        for prefix, size in (("input", 256), ("output", 10)):
            cells = net.population(prefix, size=size, neuron=lang.CustomNeuron(NEURON))
            for name, signal in (
                (prefix + "_weight", source),
                (prefix + "_bias", bias),
            ):
                projection = net.connect(
                    signal,
                    cells.excitatory,
                    name=name,
                    synapse=lang.ExponentialCurrent(tau=0.001 * lang.ms),
                    weight=lang.Constant(0),
                    initialization_scaling="direct",
                )
                self.ids[name] = projection.weight.id
            source = cells.spikes
        score = lang.ops.reduce(source, operation="sum", over="time", name="scores")
        net.output("class_scores", score)
        plan = plan_graph(lang.compile(net).graph)
        options = (
            {"projection_mode": mode}
            if "projection_mode" in inspect.signature(GraphExecutor).parameters
            else {}
        )
        self.executor = GraphExecutor(
            plan,
            seed=42,
            trainable_parameters=tuple(self.ids.values()),
            surrogate=recipe.CustomSurrogate(SURROGATE).json(),
            **options,
        )
        self.snapshots = snapshots
        self.snapshot_api = (
            "retain_runtime_state"
            in inspect.signature(self.executor.forward).parameters
        )
        restore(self, weights)

    def parameters_by_name(self):
        return {
            name: self.executor.parameter_map()[pid] for name, pid in self.ids.items()
        }

    def forward(self, events):
        options = {}
        if self.snapshot_api:
            options = dict(
                retain_parameters=self.snapshots,
                retain_final_state=self.snapshots,
                retain_runtime_state=self.snapshots,
            )
        result = self.executor(
            {
                "events": events,
                "bias": torch.ones(*events.shape[:2], 1, device=events.device),
            },
            diagnostics=False,
            **options,
        )
        return result.outputs["class_scores"]


def restore(model, weights):
    with torch.no_grad():
        for name, value in model.parameters_by_name().items():
            value.copy_(weights[name])


def copied_weights(model):
    return {
        name: value.detach().clone()
        for name, value in model.parameters_by_name().items()
    }


def encode(features, seed, steps=50):
    generator = torch.Generator(device=features.device).manual_seed(seed)
    return (
        torch.rand(steps, *features.shape, device=features.device, generator=generator)
        < features.unsqueeze(0) * 0.25
    ).float()


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def operation(model, events, labels, kind, optimizer):
    if kind == "forward":
        with torch.no_grad():
            return model(events)
    if kind == "step":
        model.zero_grad(set_to_none=True)
    scores = model(events)
    torch.nn.functional.cross_entropy(scores, labels).backward()
    if kind == "step":
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    return scores


def ablation(models, states, events, labels, repetitions):
    results = {}
    agreement = {}
    payloads = {}
    optimizers = {
        name: torch.optim.Adam(model.parameters(), lr=0.001)
        for name, model in models.items()
    }
    jobs = [(name, kind) for name in models for kind in ("forward", "backward", "step")]
    rng = random.Random(42)
    for state_name, weights in states.items():
        checked = {}
        for name, model in models.items():
            restore(model, weights)
            model.zero_grad(set_to_none=True)
            scores = operation(model, events, labels, "backward", optimizers[name])
            checked[name] = (
                scores.detach().cpu().numpy(),
                {
                    key: value.grad.detach().cpu().numpy()
                    for key, value in model.parameters_by_name().items()
                },
            )
        for name, (scores, gradients) in checked.items():
            payloads[f"{state_name}__{name}__scores"] = scores
            for key, value in gradients.items():
                payloads[f"{state_name}__{name}__{key}"] = value
        reference = checked["snntorch"]
        discrepancies = {}
        for name, (scores, gradients) in checked.items():
            np.testing.assert_array_equal(
                scores, reference[0], err_msg=state_name + "/" + name
            )
            discrepancies[name] = max(
                float(np.abs(gradients[key] - reference[1][key]).max())
                for key in gradients
            )
            for key in gradients:
                np.testing.assert_allclose(
                    gradients[key], reference[1][key], atol=1e-5, rtol=1e-4
                )
        agreement[state_name] = {
            "class_scores_exact": True,
            "gradient_atol": 1e-5,
            "gradient_rtol": 1e-4,
            "maximum_gradient_discrepancies": discrepancies,
        }
        for _ in range(3):
            rng.shuffle(jobs)
            for name, kind in jobs:
                restore(models[name], weights)
                models[name].zero_grad(set_to_none=True)
                operation(models[name], events, labels, kind, optimizers[name])
        times = {job: [] for job in jobs}
        for _ in range(repetitions):
            rng.shuffle(jobs)
            for name, kind in jobs:
                model = models[name]
                restore(model, weights)
                model.zero_grad(set_to_none=True)
                sync()
                started = time.perf_counter()
                operation(model, events, labels, kind, optimizers[name])
                sync()
                times[name, kind].append((time.perf_counter() - started) * 1000)
        rows = {}
        for name, kind in jobs:
            restore(models[name], weights)
            models[name].zero_grad(set_to_none=True)
            sync()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as profile:
                operation(models[name], events, labels, kind, optimizers[name])
                sync()
            keys = {
                "aten::mul",
                "aten::add",
                "aten::clone",
                "aten::_local_scalar_dense",
                "cudaStreamSynchronize",
            }
            rows[name + "/" + kind] = {
                "median_ms": statistics.median(times[name, kind]),
                "samples_ms": times[name, kind],
                "operators": {
                    row.key: row.count
                    for row in profile.key_averages()
                    if row.key in keys
                },
            }
        results[state_name] = rows
    return results, agreement, payloads


def pipeline(
    model, features, labels, test_features, test_labels, epochs, batch, seed=42
):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []
    sync()
    started = time.perf_counter()
    for epoch in range(epochs):
        order = torch.randperm(
            len(features), generator=torch.Generator().manual_seed(seed + epoch)
        ).to(features.device)
        losses = torch.zeros((), device=features.device, dtype=torch.float64)
        for start in range(0, len(features), batch):
            indices = order[start : start + batch]
            inputs = encode(
                features.index_select(0, indices), seed + 10000 * epoch + start
            )
            targets = labels.index_select(0, indices)
            model.zero_grad(set_to_none=True)
            scores = model(inputs)
            loss = torch.nn.functional.cross_entropy(scores, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            losses += loss.detach().double() * len(indices)
            if start % (batch * 256) == 0:
                print(f"epoch {epoch + 1}, samples {start}/{len(features)}", flush=True)
        history.append(float(losses / len(features)))
    sync()
    train_s = time.perf_counter() - started
    sync()
    started = time.perf_counter()
    correct = torch.zeros((), device=features.device, dtype=torch.long)
    loss_total = torch.zeros((), device=features.device, dtype=torch.float64)
    with torch.no_grad():
        for start in range(0, len(test_features), batch):
            events = encode(
                test_features[start : start + batch], seed + 1000000 + start
            )
            targets = test_labels[start : start + batch]
            scores = model(events)
            correct += (scores.argmax(-1) == targets).sum()
            loss_total += torch.nn.functional.cross_entropy(
                scores, targets
            ).double() * len(targets)
    sync()
    test_s = time.perf_counter() - started
    return {
        "train_s": train_s,
        "test_s": test_s,
        "train_samples": len(features),
        "test_samples": len(test_features),
        "epochs": epochs,
        "batch_size": batch,
        "epoch_losses": history,
        "test_loss": float(loss_total / len(test_features)),
        "test_accuracy": float(correct) / len(test_features),
    }, copied_weights(model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-samples", type=int, default=60000)
    parser.add_argument("--test-samples", type=int, default=10000)
    parser.add_argument("--repetitions", type=int, default=11)
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--reference-weights", type=Path)
    args = parser.parse_args()
    from torchvision.datasets import MNIST

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    train_data = MNIST("/tmp/snnlab-mnist", train=True, download=True)
    test_data = MNIST("/tmp/snnlab-mnist", train=False, download=True)
    features = (
        train_data.data[: args.train_samples].reshape(-1, 784).cuda().float() / 255
    )
    labels = train_data.targets[: args.train_samples].cuda()
    test_features = (
        test_data.data[: args.test_samples].reshape(-1, 784).cuda().float() / 255
    )
    test_labels = test_data.targets[: args.test_samples].cuda()
    weights = {name: value.cuda() for name, value in initial_weights().items()}
    reference_trained = None
    if args.reference_weights:
        with np.load(args.reference_weights) as archive:
            weights = {
                name: torch.from_numpy(archive[f"initial__{name}"].copy()).cuda()
                for name in NAMES
            }
            reference_trained = {
                name: torch.from_numpy(archive[f"trained__{name}"].copy()).cuda()
                for name in NAMES
            }
    models = {
        "snntorch": TorchModel(weights).cuda(),
        "strict_full": GraphModel(weights, "strict", True).cuda(),
    }
    if not args.baseline:
        models.update(
            fast_full=GraphModel(weights, "fast", True).cuda(),
            fast_outputs=GraphModel(weights, "fast", False).cuda(),
        )
    sync()
    setup_s = time.perf_counter() - started
    pipelines = {}
    trained = reference_trained
    for name, model in models.items() if reference_trained is None else ():
        print(f"complete pipeline: {name}", flush=True)
        restore(model, weights)
        if args.diagnostic_only:
            subset = slice(0, min(2048, len(features)))
            row, end = pipeline(
                model,
                features[subset],
                labels[subset],
                test_features[:256],
                test_labels[:256],
                1,
                128,
            )
        else:
            row, end = pipeline(
                model, features, labels, test_features, test_labels, args.epochs, 128
            )
        pipelines[name] = row
        if name == "snntorch":
            trained = end
    events = encode(features[:128], 642)
    results, agreement, payloads = ablation(
        models,
        {"initial": weights, "trained": trained},
        events,
        labels[:128],
        args.repetitions,
    )
    # A second batch size checks shape sensitivity without claiming replication.
    smaller, smaller_agreement, smaller_payloads = ablation(
        models,
        {"initial": weights, "trained": trained},
        events[:, :32],
        labels[:32],
        args.repetitions,
    )
    report = {
        "torch": str(torch.__version__),
        "snntorch": snntorch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "baseline": args.baseline,
        "setup_s": setup_s,
        "protocol": {
            "steps": 50,
            "widths": [784, 256, 10],
            "alpha": 0.8,
            "beta": 0.9,
            "dtype": "float32",
            "tf32": False,
            "cpu_threads": 4,
            "encoding": "Bernoulli(feature*0.25)",
            "warmups": 3,
            "repetitions": args.repetitions,
            "new_trained_states": True,
            "timings": "randomized interleaving; CUDA synchronized; profiling separate",
            "full_pipeline_scope": "batch selection, encoding, forward/loss/backward/clipping/Adam and test inference; acquisition/model construction excluded",
        },
        "pipelines": pipelines,
        "batch_128": results,
        "batch_32": smaller,
        "agreement": {"batch_128": agreement, "batch_32": smaller_agreement},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output.with_suffix(".weights.npz"),
        **{f"initial__{name}": value.cpu().numpy() for name, value in weights.items()},
        **{f"trained__{name}": value.cpu().numpy() for name, value in trained.items()},
    )
    np.savez(
        args.output.with_suffix(".agreement.npz"),
        **{f"batch128__{key}": value for key, value in payloads.items()},
        **{f"batch32__{key}": value for key, value in smaller_payloads.items()},
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output.read_text())


if __name__ == "__main__":
    main()
