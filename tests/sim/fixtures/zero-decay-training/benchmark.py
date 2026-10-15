"""Historical numerical helpers for sustained float32 training regressions."""

import inspect
import math

import numpy as np
import snntorch
import torch
from torch import nn

from snnlab import extensions as E
from snnlab import lang
from snnlab.sim import GraphExecutor, plan_graph

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
        score = lang.reduce(source, operation="sum", over="time", name="scores")
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
            surrogate=lang.CustomSurrogate(SURROGATE).json(),
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


def assert_training_agreement(pipelines):
    """Gate independent training separately from shared-checkpoint agreement.

    Timings are excluded. Losses must agree to 1e-5 absolute/relative and
    accuracy to one tenth of a percentage point. Failed runs retain their
    diagnostic artifacts rather than silently certifying training parity.
    """
    reference = pipelines["snntorch"]
    for name, actual in pipelines.items():
        if name == "snntorch":
            continue
        for field in ("train_samples", "test_samples", "epochs", "batch_size"):
            if actual[field] != reference[field]:
                raise AssertionError(f"training agreement: {name}/{field} differs")
        for field in ("epoch_losses", "test_loss", "test_accuracy"):
            np.testing.assert_allclose(
                actual[field],
                reference[field],
                atol=0.001 if field == "test_accuracy" else 1e-5,
                rtol=0 if field == "test_accuracy" else 1e-5,
                equal_nan=False,
                err_msg=f"independent training agreement: {name}/{field}",
            )
