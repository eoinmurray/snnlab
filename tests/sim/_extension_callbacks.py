"""Importable callbacks for cross-process extension integration tests."""

import torch


def neuron_init(ctx):
    return {
        "voltage": torch.zeros(ctx.shape, dtype=ctx.dtype, device=ctx.device),
        "adaptation": torch.zeros(ctx.shape, dtype=ctx.dtype, device=ctx.device),
    }


def neuron_step(ctx):
    voltage = (
        ctx.state["voltage"] + ctx.excitatory - ctx.inhibitory - ctx.state["adaptation"]
    )
    spikes = ctx.spike(voltage - ctx.config["threshold"])
    return {
        **ctx.state,
        "voltage": voltage - spikes,
        "adaptation": 0.5 * ctx.state["adaptation"] + 0.1 * spikes,
    }, spikes


def validate_neuron(config):
    if config["threshold"] <= 0:
        raise ValueError("threshold must be positive")


def synapse_init(ctx):
    return {
        "value": torch.zeros(ctx.shape, dtype=ctx.dtype, device=ctx.device),
        "trace": torch.zeros(ctx.shape, dtype=ctx.dtype, device=ctx.device),
    }


def synapse_step(ctx):
    trace = ctx.state["trace"] * 0.5 + ctx.drive
    return {"value": trace, "trace": trace}


def normal(shape, config, **kw):
    return torch.full(shape, config["value"], **kw)


def bounds(value, config):
    return value.clamp(-config["limit"], config["limit"])


def affine(sources, parameters, config):
    return sources[0].mean(0) * parameters["gain"]


def mse(prediction, target, config):
    return (prediction - target).square().mean()


def energy(signals, duration, config):
    return signals[0].square().mean()


def sgd(groups, config):
    return torch.optim.SGD(groups, **config)


def triangle(value, config):
    return (1 - value.abs()).clamp(min=0)


not_callable = 123
