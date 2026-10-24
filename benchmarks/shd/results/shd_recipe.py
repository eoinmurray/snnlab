"""Calibrated feedforward COBA recipe, copied from Pinglab's SHD experiment.

Only public Lang declarations and registered extension policies live here.
SNNLab's native trainer owns all simulation and optimization.
"""

from dataclasses import dataclass

import torch

from snnlab import extensions, lang
from snnlab.lang import training

TRAINABLE = (
    "input_to_E1.weight",
    "E1_to_E2.weight",
    "readout_projection.weight",
)


@dataclass(frozen=True)
class Config:
    epochs: int = 10
    batch_size: int = 256
    seed: int = 42
    steps: int = 1000
    dt_ms: float = 1.0
    channels: int = 700
    hidden: int = 128
    classes: int = 20
    validation_fraction: float = 0.1
    input_noise_hz: float = 1.6
    spike_deletion_probability: float = 0.0008

    def __post_init__(self):
        if (
            min(
                self.epochs,
                self.batch_size,
                self.steps,
                self.channels,
                self.hidden,
                self.classes,
            )
            < 1
        ):
            raise ValueError("counts must be positive")
        if self.dt_ms <= 0 or not 0 < self.validation_fraction < 1:
            raise ValueError("invalid timestep or validation fraction")
        if (
            not 0 <= self.spike_deletion_probability <= 1
            or not 0 <= self.input_noise_hz * self.dt_ms / 1000 <= 1
        ):
            raise ValueError("invalid corruption probabilities")


def equal_class_cross_entropy(prediction, target, config):
    losses = torch.nn.functional.cross_entropy(
        prediction, target.long(), reduction="none"
    )
    counts = torch.zeros(config["classes"], device=losses.device, dtype=losses.dtype)
    counts.scatter_add_(0, target, torch.ones_like(target, dtype=losses.dtype))
    return (losses / counts[target]).sum() / (counts > 0).sum()


def rate_normalized_scores(sources, parameters, config):
    scores, counts = sources
    rate = counts.mean(dim=-1, keepdim=True) / config["steps"]
    return scores * config["reference_rate"] / rate.clamp_min(config["minimum_rate"])


def centered_readout(shape, config, *, device, dtype):
    value = torch.randn(shape, device=device, dtype=dtype) * config["std"]
    return value - value.mean(dim=0, keepdim=True)


def identity_gain(shape, config, *, device, dtype):
    return torch.eye(*shape, device=device, dtype=dtype) * config["gain"]


def adam_optimizer(groups, config):
    return torch.optim.Adam(groups, **config)


extensions.register_objective(
    "benchmarks.shd.equal_class_ce/v1", equal_class_cross_entropy, classification=True
)
extensions.register_operation(
    "benchmarks.shd.rate_normalize/v1", rate_normalized_scores
)
extensions.register_initializer("benchmarks.shd.centered_readout/v1", centered_readout)
extensions.register_initializer("benchmarks.shd.identity_gain/v1", identity_gain)
extensions.register_optimizer("benchmarks.shd.adam/v1", adam_optimizer)


def author_network(cfg=Config()):
    net = lang.Network("shd_coba", dt=cfg.dt_ms * lang.ms)
    source = net.input(
        "drive",
        shape=("time", "batch", cfg.channels),
        signal_type="spikes",
        unit="spike",
    )
    learned, populations = [], []
    for index, zero_fraction in ((1, 0.95), (2, 0.9)):
        population = net.population(
            f"E{index}",
            size=cfg.hidden,
            neuron=lang.COBA_LIF(
                tau_mem=20 * lang.ms,
                capacitance_nf=1.0,
                leak_us=0.05,
                resting_mv=-65.0,
                reset_mv=-65.0,
                threshold_mv=-50.0,
                refractory=0 * lang.ms,
                voltage_grad_dampen=1.0,
            ),
        )
        projection = net.connect(
            source,
            population.excitatory,
            name="input_to_E1" if index == 1 else "E1_to_E2",
            synapse=lang.AMPA(tau=10 * lang.ms),
            weight=lang.LowerClampedNormal(
                mean=0.09, std=0.009, initial_zero_fraction=zero_fraction
            ),
            constraint=lang.NonNegative(),
            initialization_scaling="fan_in_normalized",
        )
        learned.append(projection.weight)
        populations.append(population)
        source = population.spikes
    readout = net.population(
        "readout",
        size=cfg.classes,
        spiking=False,
        neuron=lang.LeakyIntegrator(tau=20 * lang.ms),
    )
    projection = net.connect(
        source,
        readout.excitatory,
        name="readout_projection",
        synapse=lang.LeakyIntegrator(tau=10 * lang.ms),
        weight=lang.CustomInitializer("benchmarks.shd.centered_readout/v1", std=0.1),
        initialization_scaling="direct",
    )
    scores = lang.ops.reduce(
        readout.voltage, operation="mean", over="time", name="scores"
    )
    gain = net.parameter(
        "score_gain.weight",
        shape=(cfg.classes, cfg.classes),
        initializer=lang.CustomInitializer(
            "benchmarks.shd.identity_gain/v1", gain=20.0
        ),
    )
    scores = net.operation(
        "linear",
        scores,
        name="scaled_scores",
        shape=scores.shape,
        unit=scores.unit,
        signal_type="continuous",
        parameters=(gain,),
        size=cfg.classes,
        trainable=False,
    )
    counts = lang.ops.reduce(
        populations[-1].spikes,
        operation="sum",
        over="time",
        name="normalization_counts",
    )
    scores = lang.ops.custom(
        "benchmarks.shd.rate_normalize/v1",
        [scores, counts],
        name="normalized_scores",
        shape=scores.shape,
        unit="1",
        steps=cfg.steps,
        reference_rate=0.02 * cfg.dt_ms,
        minimum_rate=0.001 * cfg.dt_ms,
    )
    net.output("class_scores", scores)
    for index, population in enumerate(populations, 1):
        net.output(
            f"e{index}_counts",
            lang.ops.reduce(
                population.spikes,
                operation="sum",
                over="time",
                name=f"e{index}_count_reduction",
            ),
        )
        if index == 2:
            net.output(
                "e_counts",
                lang.ops.reduce(
                    population.spikes,
                    operation="sum",
                    over="time",
                    name="e_count_reduction",
                ),
            )
    recipe = lang.TrainSpec(
        objectives=(
            training.CustomObjective(
                "benchmarks.shd.equal_class_ce/v1",
                prediction=scores,
                target="class",
                classes=cfg.classes,
            ),
        ),
        parameter_groups=(
            training.ParameterGroup((learned[0],), name="input", lr=0.000005),
            training.ParameterGroup((learned[1],), name="interlayer", lr=0.00002),
            training.ParameterGroup((projection.weight,), name="readout", lr=0.01),
            training.ParameterGroup((gain,), name="fixed_loop", lr=0.0, frozen=True),
        ),
        optimizer=training.CustomOptimizer(
            "benchmarks.shd.adam/v1", lr=0.001, betas=[0.9, 0.999], weight_decay=0.0
        ),
        surrogate=training.FastSigmoid(
            slope=100 / 15, gradient_scale=0.01, threshold_inclusive=False
        ),
        gradient_clip=None,
        epochs=cfg.epochs,
        presentation_duration=cfg.steps * cfg.dt_ms * lang.ms,
    )
    return lang.compile(net, training=recipe)
