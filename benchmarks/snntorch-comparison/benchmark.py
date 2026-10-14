"""Matched current-based MNIST training and isolated pass measurements."""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import os
import platform
import random
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import snntorch as snn
import torch
from torch import nn
from torch.nn import functional as F

from snnlab import __version__ as snnlab_version
from snnlab import extensions as E
from snnlab import lang
from snnlab.lang import training as recipe
from snnlab.sim.execution import (
    DenseArrayBinding,
    ExecutionSpec,
    GraphExecutor,
    ValidationSpec,
    plan_graph,
    save_training_checkpoint,
    train,
)

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REQUIRED_RELEASE = "0.12.1"
EXECUTION_POLICY = {
    "projection_mode": "fast",
    "extension_validation": "structural",
    "compile_mode": None,
    "diagnostics": False,
    "retain_parameters": False,
    "retain_final_state": False,
    "retain_runtime_state": False,
}


@dataclass(frozen=True)
class Config:
    seed: int = 17
    train_size: int = 6000
    test_size: int = 10000
    hidden: int = 1000
    steps: int = 25
    batch_size: int = 128
    epochs: int = 20
    alpha: float = 0.9
    beta: float = 0.95
    lr: float = 5e-4

    def __post_init__(self):
        for name in (
            "train_size",
            "test_size",
            "hidden",
            "steps",
            "batch_size",
            "epochs",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= self.alpha < 1 or not 0 < self.beta < 1 or self.lr <= 0:
            raise ValueError("require 0 <= alpha < 1, 0 < beta < 1 and lr > 0")
        if self.train_size > 60000 or self.test_size > 10000:
            raise ValueError("requested split exceeds MNIST")

    def json(self):
        return asdict(self)


def source_manifest():
    paths = sorted((ROOT / "src").rglob("*.py")) + sorted(HERE.glob("*.py"))
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_identity():
    """The uploaded source is authoritative; the commit identifies its base."""
    commit = os.environ.get("SNNLAB_BENCHMARK_COMMIT")
    if commit is None:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        commit = result.stdout.strip() if result.returncode == 0 else None
    return {"commit": commit, "snnlab_version": snnlab_version}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def configure(device):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    if torch.device(device).type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def require_optimized_runtime():
    """Reject older executors rather than silently retaining expensive snapshots."""
    if tuple(map(int, snnlab_version.split("."))) < tuple(map(int, REQUIRED_RELEASE.split("."))):
        raise RuntimeError(
            f"This run requires SNNLab >= {REQUIRED_RELEASE}; found {snnlab_version}."
        )
    constructor = inspect.signature(GraphExecutor).parameters
    forward = inspect.signature(GraphExecutor.forward).parameters
    request = inspect.signature(ExecutionSpec).parameters
    if "training_timing" not in request or "projection_mode" not in constructor or not all(
        name in forward
        for name in ("retain_parameters", "retain_final_state", "retain_runtime_state")
    ):
        raise RuntimeError(
            "This benchmark requires projection/snapshot controls and training_timing support."
        )


def initialize(ctx):
    return {
        "voltage": torch.zeros(ctx.shape, device=ctx.device, dtype=ctx.dtype),
        "refractory": torch.zeros(ctx.shape, device=ctx.device, dtype=torch.long),
    }


def current_step(ctx):
    """Delayed subtractive reset with a detached reset decision, as in Tutorial 5.

    Incoming current already includes the projection's exponential filter.
    Normalized units assign 1 mV per nA per step; rest is zero and threshold 1 mV.
    """
    previous = ctx.state["voltage"]
    reset = (previous > 1.0).to(previous.dtype).detach()
    voltage = ctx.config["beta"] * previous + ctx.excitatory - ctx.inhibitory - reset
    return {**ctx.state, "voltage": voltage}, ctx.spike(voltage - 1.0)


def atan_derivative(value, config):
    # snnTorch's default ATan surrogate has alpha=2.
    return 1.0 / (1.0 + (math.pi * value).square())


def matched_initializer(shape, config, *, device, dtype):
    weights = initial_weights(Config(seed=config["seed"], hidden=config["hidden"]))
    values = {
        "fc1": torch.cat([weights["fc1.weight"].T, weights["fc1.bias"][None]], dim=0),
        "fc2": weights["fc2.weight"].T,
        "output_bias": weights["fc2.bias"][None],
    }
    value = values[config["projection"]].to(device=device, dtype=dtype)
    if tuple(value.shape) != tuple(shape):
        raise ValueError("matched initializer shape mismatch")
    return value


def temporal_objective(prediction, target, config):
    return loss_fn(prediction, target)


def accuracy_objective(prediction, target, config):
    # A reporting-only classification objective: no loss or gradient contribution.
    return prediction.new_zeros(())


def adam_factory(groups, config):
    return torch.optim.Adam(groups, **config)


def initial_weights(config):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        fc1 = nn.Linear(784, config.hidden)
        fc2 = nn.Linear(config.hidden, 10)
    return {
        "fc1.weight": fc1.weight.detach().clone(),
        "fc1.bias": fc1.bias.detach().clone(),
        "fc2.weight": fc2.weight.detach().clone(),
        "fc2.bias": fc2.bias.detach().clone(),
    }


class TorchNetwork(nn.Module):
    """snnTorch's own Synaptic neurons, default surrogate and reset semantics."""

    def __init__(self, config, weights):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(784, config.hidden)
        self.fc2 = nn.Linear(config.hidden, 10)
        self.hidden = snn.Synaptic(alpha=config.alpha, beta=config.beta)
        self.output = snn.Synaptic(alpha=config.alpha, beta=config.beta)
        for layer, name in ((self.fc1, "fc1"), (self.fc2, "fc2")):
            with torch.no_grad():
                layer.weight.copy_(weights[f"{name}.weight"])
                layer.bias.copy_(weights[f"{name}.bias"])

    def forward(self, images):
        h_syn = images.new_zeros(images.shape[0], self.config.hidden)
        h_mem = torch.zeros_like(h_syn)
        o_syn = images.new_zeros(images.shape[0], 10)
        o_mem = torch.zeros_like(o_syn)
        spikes, voltage = [], []
        for _ in range(self.config.steps):
            h_spike, h_syn, h_mem = self.hidden(self.fc1(images), h_syn, h_mem)
            o_spike, o_syn, o_mem = self.output(self.fc2(h_spike), o_syn, o_mem)
            spikes.append(o_spike)
            voltage.append(o_mem)
        return torch.stack(spikes), torch.stack(voltage)

    def canonical_parameters(self):
        return {
            "fc1.weight": self.fc1.weight,
            "fc1.bias": self.fc1.bias,
            "fc2.weight": self.fc2.weight,
            "fc2.bias": self.fc2.bias,
        }


class LabNetwork(nn.Module):
    """Public SNNLab authoring, exponential-current projections and executor."""

    def __init__(self, config, weights):
        super().__init__()
        self.config = config
        require_optimized_runtime()
        definitions = {definition.name for definition in E.definitions()}
        if "comparison.tutorial_current/v1" not in definitions:
            E.register_neuron(
                "comparison.tutorial_current/v1", current_step, initialize=initialize
            )
        if "comparison.atan/v1" not in definitions:
            E.register_surrogate("comparison.atan/v1", atan_derivative)
        extensions = (
            ("initializer", "comparison.weights/v1", matched_initializer, {}),
            ("objective", "comparison.temporal_ce/v1", temporal_objective, {}),
            (
                "objective",
                "comparison.spike_accuracy/v1",
                accuracy_objective,
                {"classification": True},
            ),
            ("optimizer", "comparison.adam/v1", adam_factory, {}),
        )
        for category, name, function, options in extensions:
            if name not in definitions:
                getattr(E, f"register_{category}")(name, function, **options)
        net = lang.Network("mnist_comparison", dt=1 * lang.ms)
        pixels = net.input(
            "pixels", shape=("time", "batch", 785), signal_type="current", unit="nA"
        )
        bias = net.input(
            "bias", shape=("time", "batch", 1), signal_type="current", unit="nA"
        )
        hidden = net.population(
            "hidden",
            size=config.hidden,
            neuron=lang.CustomNeuron(
                "comparison.tutorial_current/v1", beta=config.beta
            ),
        )
        output = net.population(
            "output",
            size=10,
            neuron=lang.CustomNeuron(
                "comparison.tutorial_current/v1", beta=config.beta
            ),
        )
        # alpha=0 is the Tutorial 5 instantaneous-current limit; exponential
        # decay underflows to zero at this tau. The default alpha=0.9 is nonzero.
        tau = -1.0 / math.log(config.alpha) if config.alpha else 1e-6
        for name, source, target in (
            ("fc1", pixels, hidden.excitatory),
            ("fc2", hidden.spikes, output.excitatory),
            ("output_bias", bias, output.excitatory),
        ):
            net.connect(
                source,
                target,
                name=name,
                synapse=lang.ExponentialCurrent(tau=tau * lang.ms),
                weight=lang.CustomInitializer(
                    "comparison.weights/v1",
                    seed=config.seed,
                    hidden=config.hidden,
                    projection=name,
                ),
                initialization_scaling="direct",
                delay=0 * lang.ms,
            )
        net.output("spikes", output.spikes)
        net.output("voltage", output.voltage)
        counts = lang.ops.reduce(
            output.spikes, operation="sum", over="time", name="spike_counts"
        )
        net.output("class_scores", counts)
        training = lang.TrainSpec(
            objectives=(
                recipe.CustomObjective(
                    "comparison.temporal_ce/v1",
                    prediction=output.voltage,
                    target="class",
                ),
                recipe.CustomObjective(
                    "comparison.spike_accuracy/v1", prediction=counts, target="class"
                ),
            ),
            parameter_groups=(
                recipe.ParameterGroup(
                    tuple(p["id"] for p in net.parameters), name="weights", lr=config.lr
                ),
            ),
            optimizer=recipe.CustomOptimizer("comparison.adam/v1", betas=(0.9, 0.999)),
            surrogate=recipe.CustomSurrogate("comparison.atan/v1"),
            presentation_duration=config.steps * lang.ms,
        )
        self.bundle = lang.compile(net, training=training)
        self.graph = self.bundle.graph
        self.executor = GraphExecutor(
            plan_graph(self.graph),
            trainable_parameters=tuple(p["id"] for p in net.parameters),
            surrogate=recipe.CustomSurrogate("comparison.atan/v1").json(),
            extension_validation=EXECUTION_POLICY["extension_validation"],
            projection_mode=EXECUTION_POLICY["projection_mode"],
            # Prepared built-in kernels can compile; this benchmark's matched
            # custom neuron/surrogate callbacks retain eager dispatch.
            compile_mode=EXECUTION_POLICY["compile_mode"],
        )
        with torch.no_grad():
            p = self.executor.parameter_map()
            p["fc1.weight"].copy_(
                torch.cat([weights["fc1.weight"].T, weights["fc1.bias"][None]], dim=0)
            )
            p["fc2.weight"].copy_(weights["fc2.weight"].T)
            p["output_bias.weight"].copy_(weights["fc2.bias"][None])

    def forward(self, images):
        augmented = torch.cat([images, images.new_ones(images.shape[0], 1)], dim=1)
        result = self.executor(
            {
                "pixels": augmented.unsqueeze(0).expand(self.config.steps, -1, -1),
                "bias": images.new_ones(self.config.steps, images.shape[0], 1),
            },
            **{
                key: EXECUTION_POLICY[key]
                for key in (
                    "diagnostics",
                    "retain_parameters",
                    "retain_final_state",
                    "retain_runtime_state",
                )
            },
        )
        return result.outputs["spikes"], result.outputs["voltage"]

    def canonical_parameters(self):
        p = self.executor.parameter_map()
        return {
            "fc1.weight": p["fc1.weight"][:-1].T,
            "fc1.bias": p["fc1.weight"][-1],
            "fc2.weight": p["fc2.weight"].T,
            "fc2.bias": p["output_bias.weight"][0],
        }

    def canonical_gradients(self):
        p = self.executor.parameter_map()
        return {
            "fc1.weight": p["fc1.weight"].grad[:-1].T,
            "fc1.bias": p["fc1.weight"].grad[-1],
            "fc2.weight": p["fc2.weight"].grad.T,
            "fc2.bias": p["output_bias.weight"].grad[0],
        }


def loss_fn(voltage, labels):
    """Tutorial 5: sum over time of batch-mean membrane cross-entropy."""
    steps, batch, classes = voltage.shape
    return (
        F.cross_entropy(voltage.reshape(steps * batch, classes), labels.repeat(steps))
        * steps
    )


def verify_pair(config, weights, device="cpu"):
    """Fail closed on spikes, voltages, loss, BPTT, and one Adam update."""
    configure(device)
    images = torch.rand(8, 784, generator=torch.Generator().manual_seed(103)).to(device)
    labels = (torch.arange(8) % 10).to(device)
    models = [
        TorchNetwork(config, weights).to(device),
        LabNetwork(config, weights).to(device),
    ]
    outputs, gradients, input_gradients = [], [], []
    for model in models:
        model_images = images.detach().clone().requires_grad_()
        spikes, voltage = model(model_images)
        loss = loss_fn(voltage, labels)
        loss.backward()
        outputs.append((spikes.detach(), voltage.detach(), loss.detach()))
        gradients.append(
            model.canonical_gradients()
            if isinstance(model, LabNetwork)
            else {k: v.grad for k, v in model.canonical_parameters().items()}
        )
        input_gradients.append(model_images.grad)
    torch.testing.assert_close(outputs[0][0], outputs[1][0], atol=0, rtol=0)
    for index in (1, 2):
        torch.testing.assert_close(
            outputs[0][index], outputs[1][index], atol=2e-5, rtol=2e-5
        )
    gradient_errors = {}
    # Bias is fused inside nn.Linear and is a separate current projection in
    # SNNLab. Float32 cancellation makes pointwise relative error unstable near
    # zero. Bound both the relative L2 error and the globally scaled max error.
    for key, reference, actual in [
        (key, gradients[0][key], gradients[1][key]) for key in gradients[0]
    ] + [("inputs", input_gradients[0], input_gradients[1])]:
        error = (actual - reference).abs()
        relative_l2 = float(error.norm() / reference.norm().clamp(min=1e-12))
        scaled_max = float(error.max() / reference.abs().max().clamp(min=1.0))
        if not math.isfinite(relative_l2) or relative_l2 > 1e-4 or scaled_max > 1e-4:
            raise AssertionError(
                f"gradient mismatch for {key}: {relative_l2=}, {scaled_max=}"
            )
        gradient_errors[key] = {"relative_l2": relative_l2, "scaled_max": scaled_max}
    for model in models:
        torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999)).step()
    for key, value in models[0].canonical_parameters().items():
        torch.testing.assert_close(
            value, models[1].canonical_parameters()[key], atol=2e-5, rtol=2e-4
        )
    return {
        "passed": True,
        "device": str(device),
        "checks": [
            "spikes",
            "voltage",
            "loss",
            "input_gradients",
            "parameter_gradients",
            "adam_update",
        ],
        "gradient_errors": gradient_errors,
        "snnlab_execution": EXECUTION_POLICY.copy(),
    }


def stratified_indices(labels, count, seed):
    """Select exactly count items, proportionally by class, without replacement."""
    generator = torch.Generator().manual_seed(seed)
    classes, sizes = torch.unique(labels, return_counts=True)
    quotas = sizes.double() * count / len(labels)
    allocation = quotas.floor().long()
    remainder = count - int(allocation.sum())
    allocation[torch.argsort(quotas - allocation, descending=True)[:remainder]] += 1
    selected = []
    for label, size in zip(classes, allocation):
        candidates = torch.where(labels == label)[0]
        selected.append(
            candidates[torch.randperm(len(candidates), generator=generator)[:size]]
        )
    result = torch.cat(selected)
    return result[torch.randperm(len(result), generator=generator)]


def epoch_order(count, seed, epoch):
    return torch.randperm(count, generator=torch.Generator().manual_seed(seed + epoch))


def prepare(config, data_root, output):
    from torchvision.datasets import MNIST

    train = MNIST(data_root, train=True, download=True)
    test = MNIST(data_root, train=False, download=True)
    train_indices = stratified_indices(train.targets, config.train_size, config.seed)
    test_indices = stratified_indices(test.targets, config.test_size, config.seed + 1)
    packet = {
        "train_images": train.data[train_indices].flatten(1).float() / 255,
        "train_labels": train.targets[train_indices],
        "test_images": test.data[test_indices].flatten(1).float() / 255,
        "test_labels": test.targets[test_indices],
        "train_indices": train_indices,
        "test_indices": test_indices,
        "weights": initial_weights(config),
        "config": config.json(),
        "synthetic": False,
        "provenance": {
            "dataset": "MNIST official train/test splits",
            "train_class_counts": torch.bincount(
                train.targets[train_indices], minlength=10
            ).tolist(),
            "train_index_digest": digest(train_indices.tolist()),
            "test_index_digest": digest(test_indices.tolist()),
            "tutorial": "https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html",
        },
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(packet, output)
    return packet


def synthetic_packet(config):
    generator = torch.Generator().manual_seed(config.seed)
    return {
        "train_images": torch.rand(config.train_size, 784, generator=generator),
        "train_labels": torch.arange(config.train_size) % 10,
        "test_images": torch.rand(config.test_size, 784, generator=generator),
        "test_labels": torch.arange(config.test_size) % 10,
        "weights": initial_weights(config),
        "config": config.json(),
        "synthetic": True,
        "provenance": {"dataset": "synthetic smoke inputs"},
    }


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def run_lab_training(
    model, config, tensors, metrics, checkpoint_path, device, on_epoch
):
    """One public sim.train request owns all epochs and optimizer updates."""

    def bindings(images):
        augmented = torch.cat([images, images.new_ones(len(images), 1)], dim=1)
        return (
            DenseArrayBinding(
                input_id="pixels",
                value=augmented.unsqueeze(0).expand(config.steps, -1, -1),
            ),
            DenseArrayBinding(
                input_id="bias", value=images.new_ones(config.steps, len(images), 1)
            ),
        )

    execution = ExecutionSpec(
        kind="train",
        graph=model.bundle.graph,
        training=model.bundle.training,
        input_bindings=bindings(tensors["train_images"]),
        targets={"class": tensors["train_labels"]},
        validation=ValidationSpec(
            input_bindings=bindings(tensors["test_images"]),
            targets={"class": tensors["test_labels"]},
        ),
        seed=config.seed,
        device=str(device),
        epochs=config.epochs,
        batch_size=config.batch_size,
        shuffle=True,
        training_timing=True,
        **EXECUTION_POLICY,
    )
    synchronize(device)
    started = time.perf_counter()
    result = train(execution)
    synchronize(device)
    elapsed = time.perf_counter() - started
    expected_updates = config.epochs * math.ceil(config.train_size / config.batch_size)
    if result.training_checkpoint.completed_updates != expected_updates:
        raise RuntimeError("native training did not complete all requested updates")
    if [row["epoch"] for row in result.metrics["epochs"]] != list(
        range(config.epochs + 1)
    ):
        raise RuntimeError("native training returned incomplete epoch evaluations")
    if not all(math.isfinite(row["loss"]) for row in result.metrics["updates"]):
        raise RuntimeError("nonfinite native training loss")
    metrics["training_api"] = "snnlab.sim.execution.train"
    metrics["train_call_seconds"] = elapsed
    metrics["wall_seconds"] = elapsed
    metrics["timing_scope"] = (
        "setup, training, initial/epoch train/test evaluations and in-memory checkpoint construction; excludes warmup, disk writes and Modal commits"
    )
    metrics["training_timing"] = result.metrics["training_timing"]
    epoch_timings = {
        row["epoch"]: row for row in result.metrics["training_timing"]["epochs"]
    }
    metrics["initial_evaluation"] = result.metrics["epochs"][0]
    metrics["epochs"] = [
        {
            "epoch": row["epoch"],
            "train_loss": row["train_loss"],
            "train_accuracy": row["train_accuracy"],
            "test_loss": row["validation_loss"],
            "test_accuracy": row["validation_accuracy"],
            "train_examples": config.train_size,
            "test_examples": config.test_size,
            "train_seconds": epoch_timings[row["epoch"]]["training_s"],
            "train_evaluation_seconds": epoch_timings[row["epoch"]][
                "train_evaluation_s"
            ],
            "test_seconds": epoch_timings[row["epoch"]]["validation_s"],
            "evaluation_seconds": epoch_timings[row["epoch"]]["train_evaluation_s"]
            + epoch_timings[row["epoch"]]["validation_s"],
        }
        for row in result.metrics["epochs"]
        if row["epoch"] > 0
    ]
    metrics["updates"] = [
        {"update": row["update"], "epoch": row["epoch"], "train_loss": row["loss"]}
        for row in result.metrics["updates"]
    ]
    metrics["epoch_metric_scope"] = "fixed weights after each completed epoch"
    if checkpoint_path:
        native_path = checkpoint_path.with_suffix(".native")
        save_training_checkpoint(native_path, result.training_checkpoint)
        p = result.training_checkpoint.parameters
        torch.save(
            {
                "config": config.json(),
                "epoch": config.epochs,
                "parameters": {
                    "fc1.weight": p["fc1.weight"][:-1].T.cpu(),
                    "fc1.bias": p["fc1.weight"][-1].cpu(),
                    "fc2.weight": p["fc2.weight"].T.cpu(),
                    "fc2.bias": p["output_bias.weight"][0].cpu(),
                },
                "source_identity": metrics["source_identity"],
                "source_manifest": metrics["source_manifest"],
                "snnlab_execution": metrics["snnlab_execution"],
                "optimizer": result.optimizer_state,
                "native_checkpoint": native_path.name,
            },
            checkpoint_path,
        )
    metrics["complete"] = True
    metrics["peak_allocated_bytes"] = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    )
    for row in metrics["epochs"]:
        print(json.dumps({"backend": "snnlab", **row}), flush=True)
    if on_epoch is not None:
        on_epoch()
    return metrics


def run_training(backend, config, packet, output=None, device="cpu", on_epoch=None):
    device = torch.device(device)
    configure(device)
    if packet["config"] != config.json():
        raise ValueError("prepared data/config mismatch")
    checkpoint_path = Path(output) if output else None
    if checkpoint_path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    verification = verify_pair(config, packet["weights"], device)
    network = {"snntorch": TorchNetwork, "snnlab": LabNetwork}[backend]
    model = network(config, packet["weights"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999))
    tensors = {
        key: packet[key].to(device)
        for key in ("train_images", "train_labels", "test_images", "test_labels")
    }
    # Identical ordering is independent of model construction and global RNG.
    orders = [
        epoch_order(config.train_size, config.seed, epoch).to(device)
        for epoch in range(config.epochs)
    ]
    synchronize(device)
    warm_start = time.perf_counter()
    for _ in range(2):
        _, voltage = model(tensors["train_images"][: config.batch_size])
        loss_fn(voltage, tensors["train_labels"][: config.batch_size]).backward()
        optimizer.zero_grad(set_to_none=True)
    synchronize(device)
    warm_seconds = time.perf_counter() - warm_start
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    metrics = {
        "backend": backend,
        "config": config.json(),
        "synthetic": packet["synthetic"],
        "provenance": packet["provenance"],
        "source_manifest": source_manifest(),
        "source_identity": source_identity(),
        "snnlab_execution": EXECUTION_POLICY.copy(),
        "verification": verification,
        "warmup_seconds": warm_seconds,
        "environment": {
            "torch": str(torch.__version__),
            "snntorch": str(snn.__version__),
            "python": platform.python_version(),
            "modal_task_id": os.environ.get("MODAL_TASK_ID"),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
        },
        "epochs": [],
        "updates": [],
        "complete": False,
    }
    if backend == "snnlab":
        for name, expected in initial_weights(config).items():
            torch.testing.assert_close(
                packet["weights"][name].cpu(), expected, atol=0, rtol=0
            )
        return run_lab_training(
            model, config, tensors, metrics, checkpoint_path, device, on_epoch
        )
    metrics["training_api"] = "torch.optim.Adam"
    metrics["epoch_metric_scope"] = "fixed weights after each completed epoch"
    metrics["timing_scope"] = (
        "setup, training, initial/epoch train/test evaluations and in-memory checkpoint construction; excludes warmup, disk writes and Modal commits"
    )
    synchronize(device)
    started = time.perf_counter()
    # Rebuild after warmup to include construction, as the native train() call does.
    model = TorchNetwork(config, packet["weights"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999))
    orders = [
        epoch_order(config.train_size, config.seed, epoch).to(device)
        for epoch in range(config.epochs)
    ]
    synchronize(device)
    timings = {
        key: 0.0
        for key in (
            "setup_s",
            "training_s",
            "train_evaluation_s",
            "validation_s",
            "checkpoint_s",
            "reporting_s",
        )
    }
    timings["setup_s"] = time.perf_counter() - started
    epoch_timings = []

    def evaluate(split):
        synchronize(device)
        begin = time.perf_counter()
        losses, correct = [], []
        model.eval()
        images, labels = tensors[f"{split}_images"], tensors[f"{split}_labels"]
        with torch.no_grad():
            for start in range(0, len(images), config.batch_size):
                batch_images, batch_labels = (
                    images[start : start + config.batch_size],
                    labels[start : start + config.batch_size],
                )
                spikes, voltage = model(batch_images)
                losses.append(loss_fn(voltage, batch_labels) * len(batch_labels))
                correct.append((spikes.sum(0).argmax(1) == batch_labels).sum())
        loss = float(torch.stack(losses).sum().cpu()) / len(images)
        accuracy = int(torch.stack(correct).sum().cpu()) / len(images)
        synchronize(device)
        return loss, accuracy, time.perf_counter() - begin

    initial_train_loss, initial_train_accuracy, initial_train_s = evaluate("train")
    initial_test_loss, initial_test_accuracy, initial_test_s = evaluate("test")
    metrics["initial_evaluation"] = {
        "epoch": 0,
        "train_loss": initial_train_loss,
        "train_accuracy": initial_train_accuracy,
        "validation_loss": initial_test_loss,
        "validation_accuracy": initial_test_accuracy,
    }
    timings["train_evaluation_s"] += initial_train_s
    timings["validation_s"] += initial_test_s
    epoch_timings.append(
        {
            "epoch": 0,
            "training_s": 0.0,
            "train_evaluation_s": initial_train_s,
            "validation_s": initial_test_s,
        }
    )
    for epoch, order in enumerate(orders):
        model.train()
        synchronize(device)
        train_start = time.perf_counter()
        batch_losses = []
        for indices in order.split(config.batch_size):
            images, labels = (
                tensors["train_images"][indices],
                tensors["train_labels"][indices],
            )
            optimizer.zero_grad(set_to_none=True)
            _, voltage = model(images)
            loss = loss_fn(voltage, labels)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.detach())
        losses = torch.stack(batch_losses).cpu().tolist()
        for loss in losses:
            metrics["updates"].append(
                {
                    "update": len(metrics["updates"]) + 1,
                    "epoch": epoch + 1,
                    "train_loss": loss,
                }
            )
        synchronize(device)
        train_s = time.perf_counter() - train_start
        train_loss, train_accuracy, train_eval_s = evaluate("train")
        test_loss, test_accuracy, test_s = evaluate("test")
        timings["training_s"] += train_s
        timings["train_evaluation_s"] += train_eval_s
        timings["validation_s"] += test_s
        epoch_timings.append(
            {
                "epoch": epoch + 1,
                "training_s": train_s,
                "train_evaluation_s": train_eval_s,
                "validation_s": test_s,
            }
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "train_seconds": train_s,
            "train_evaluation_seconds": train_eval_s,
            "test_seconds": test_s,
            "evaluation_seconds": train_eval_s + test_s,
            "cumulative_train_seconds": timings["training_s"],
            "train_examples": config.train_size,
            "test_examples": config.test_size,
        }
        if not all(math.isfinite(row[key]) for key in ("train_loss", "test_loss")):
            raise RuntimeError("nonfinite epoch loss")
        metrics["epochs"].append(row)
    synchronize(device)
    checkpoint_start = time.perf_counter()
    checkpoint = {
        "config": config.json(),
        "epoch": config.epochs,
        "parameters": {
            key: value.detach().clone()
            for key, value in model.canonical_parameters().items()
        },
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "source_identity": metrics["source_identity"],
        "source_manifest": metrics["source_manifest"],
        "snnlab_execution": metrics["snnlab_execution"],
    }
    synchronize(device)
    timings["checkpoint_s"] = time.perf_counter() - checkpoint_start
    metrics["wall_seconds"] = time.perf_counter() - started
    timings["reporting_s"] = metrics["wall_seconds"] - sum(timings.values())
    metrics["training_timing"] = {
        "mode": "synchronized",
        "total_s": metrics["wall_seconds"],
        **timings,
        "epochs": epoch_timings,
    }
    # Persist and commit only after the matched timing boundary.
    if checkpoint_path:
        checkpoint["parameters"] = {
            key: value.cpu() for key, value in checkpoint["parameters"].items()
        }
        torch.save(checkpoint, checkpoint_path)
    metrics["complete"] = True
    metrics["peak_allocated_bytes"] = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    )
    for row in metrics["epochs"]:
        print(json.dumps({"backend": backend, **row}), flush=True)
    if on_epoch is not None:
        on_epoch()
    return metrics


def timed_pass(model, images, labels, kind):
    """Fresh autograd graph for each backward; preparation is outside timing."""
    device = images.device
    model.zero_grad(set_to_none=True)
    model.train(kind != "inference_forward")
    if kind == "backward":
        _, voltage = model(images)
        loss = loss_fn(voltage, labels)
    elif kind not in ("training_forward", "inference_forward"):
        raise ValueError(f"unknown timing kind: {kind}")
    synchronize(device)
    started = time.perf_counter()
    if kind == "backward":
        loss.backward()
        output = None
    else:
        with torch.set_grad_enabled(kind == "training_forward"):
            output = model(images)
    synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000
    # Keep outputs alive until after the end fence: destruction is not a pass.
    del output
    return elapsed_ms


def measure_passes(models, images, labels, *, warmups=5, repetitions=50, seed=17):
    if warmups < 0 or repetitions < 2:
        raise ValueError("require nonnegative warmups and at least two repetitions")
    kinds = ("training_forward", "backward", "inference_forward")
    jobs = [(backend, kind) for backend in models for kind in kinds]
    rng = random.Random(seed)
    for _ in range(warmups):
        rng.shuffle(jobs)
        for backend, kind in jobs:
            timed_pass(models[backend], images, labels, kind)
    samples = {job: [] for job in jobs}
    for _ in range(repetitions):
        rng.shuffle(jobs)
        for backend, kind in jobs:
            samples[backend, kind].append(
                timed_pass(models[backend], images, labels, kind)
            )
    report = {backend: {} for backend in models}
    for (backend, kind), values in samples.items():
        quartiles = statistics.quantiles(values, n=4, method="inclusive")
        report[backend][kind] = {
            "median_ms": statistics.median(values),
            "mean_ms": statistics.mean(values),
            "q1_ms": quartiles[0],
            "q3_ms": quartiles[2],
            "samples_ms": values,
        }
    return report


def compare_outputs(models, images, labels):
    """Record numerical agreement and reject nonfinite outputs/gradients."""
    outputs = {}
    for backend, model in models.items():
        model.zero_grad(set_to_none=True)
        spikes, voltage = model(images)
        loss = loss_fn(voltage, labels)
        loss.backward()
        values = [spikes, voltage, loss] + [
            parameter.grad for parameter in model.parameters()
        ]
        if any(value is None or not torch.isfinite(value).all() for value in values):
            raise RuntimeError(f"nonfinite or missing value in {backend} preflight")
        outputs[backend] = (spikes.detach(), voltage.detach(), loss.detach())
    left, right = outputs["snntorch"], outputs["snnlab"]
    return {
        "finite_outputs_and_gradients": True,
        "different_output_spikes": int((left[0] != right[0]).sum()),
        "max_voltage_error": float((left[1] - right[1]).abs().max()),
        "loss_absolute_error": float((left[2] - right[2]).abs()),
    }


def combined_results(training, *, run_id="local", cloud=None, profiling=None):
    left, right = training["snntorch"], training["snnlab"]
    shared = (
        "config",
        "provenance",
        "source_manifest",
        "source_identity",
        "snnlab_execution",
        "synthetic",
    )
    for key in shared:
        if left[key] != right[key]:
            raise ValueError(f"comparison mismatch: {key}")
    if not all(
        item["complete"] and item["verification"]["passed"]
        for item in training.values()
    ):
        raise ValueError("both verified training jobs must complete")
    return {
        "schema": 1,
        "run_id": run_id,
        "protocol": {key: left[key] for key in shared},
        "training": training,
        "profiling": profiling,
        "cloud": cloud or {},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = (
        Config()
        if args.verify
        else Config(
            train_size=32, test_size=16, hidden=16, steps=8, batch_size=16, epochs=2
        )
    )
    if args.verify:
        print(json.dumps(verify_pair(config, initial_weights(config)), indent=2))
    else:
        if args.output is None:
            parser.error("--smoke requires --output (use a temporary directory)")
        packet = synthetic_packet(config)
        training = {
            name: run_training(name, config, packet) for name in ("snntorch", "snnlab")
        }
        write_json(args.output / "results.json", combined_results(training))
        from plot import make_plots

        make_plots(args.output)


if __name__ == "__main__":
    main()
