"""Encoders that turn images into spike trains.

Everything here is pure-data — no model state, no CLI plumbing. Lifted out
of cli.py.
"""

from __future__ import annotations

import inspect

import torch

EVAL_SEED = 20260415


def transfer_compact(value, device):
    """Stage compact rates/coordinates without a blocking CUDA host wait."""
    device = torch.device(device)
    if device.type == "cuda" and value.device.type == "cpu":
        return value.pin_memory().to(device, non_blocking=True)
    return value.to(device)


def accepts_device(function):
    """An explicit optional device keyword opts a registered encoder in."""
    try:
        parameter = inspect.signature(function).parameters.get("device")
    except (TypeError, ValueError):
        return False
    return parameter is not None and parameter.kind in {
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }


def encoder_device_policy(device):
    """Retain CPU encoder identities; bind accelerator callback preparation."""
    backend = torch.device(device).type
    return (
        {
            "preparation_device": backend,
            "preparation_policy": "snnlab.encoder-device/v1",
            "preparation_torch_version": str(torch.__version__),
        }
        if backend != "cpu"
        else {}
    )


def poisson_device_policy(device):
    """Record device RNG changes without changing historical CPU identities."""
    backend = torch.device(device).type
    return (
        {
            "random_policy": {
                "schema": "snnlab.poisson-device/v1",
                "backend": backend,
                "random_dtype": "float32",
                "torch_version": str(torch.__version__),
                "rate_selection": "cpu",
                "cross_device_bitwise": False,
            }
        }
        if backend != "cpu"
        else {}
    )


def encode_images_poisson(images, T_steps, dt, max_rate_hz, generator=None):
    """Encode (B, N_in) pixel intensities as Poisson spike trains.

    Returns (T_steps, B, N_in) float spikes. Single canonical encoder used by
    train, infer, and image paths so identical pixels with the same dt
    and max_rate produce the same spike train regardless of mode.
    """
    pixels = images.clamp(0, 1)
    B, n_in = pixels.shape
    rates = torch.as_tensor(max_rate_hz, dtype=pixels.dtype, device=pixels.device)
    if rates.ndim == 0:
        rates = rates.expand(B)
    if rates.shape != (B,):
        raise ValueError(
            f"max_rate_hz must be scalar or shape ({B},), got {tuple(rates.shape)}"
        )
    if torch.any(rates < 0):
        raise ValueError("input rates must be non-negative")
    p = rates.reshape(1, B, 1) * dt / 1000.0
    if torch.any(p > 1):
        raise ValueError("input rate and dt produce Bernoulli probability above one")
    if generator is not None:
        # Generator dictates device (usually CPU); generate there then move.
        rand = torch.rand(
            T_steps, B, n_in, device=generator.device, generator=generator
        ).to(pixels.device)
    else:
        rand = torch.rand(T_steps, B, n_in, device=pixels.device)
    return (rand < pixels.unsqueeze(0) * p).float()
