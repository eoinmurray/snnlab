"""Shared-kernel connectivity and spatial operations over flat signals."""

from __future__ import annotations

import math

from snnlab._shared.spatial import geometry, pair, signal_spatial_shape
from snnlab.lang._specs import Spec
from snnlab.lang.network import Signal


def Conv2d(*, kernel_size, stride=1, padding=0) -> Spec:
    """Cross-correlation connectivity, independent of synaptic dynamics."""
    return Spec(
        "conv2d",
        {
            "kernel_size": list(pair(kernel_size, name="kernel_size")),
            "stride": list(pair(stride, name="stride")),
            "padding": list(pair(padding, name="padding", minimum=0)),
        },
    )


def pool2d(
    source: Signal, *, kernel_size, name: str, stride=None, padding=0, mode="avg"
) -> Signal:
    """Spatial pooling; average results are continuous, max preserves spikes."""
    if mode not in {"avg", "max"}:
        raise ValueError("pooling mode must be 'avg' or 'max'")
    net = source.network
    shape = signal_spatial_shape(vars(net), source.id)
    config = Conv2d(
        kernel_size=kernel_size,
        stride=kernel_size if stride is None else stride,
        padding=padding,
    ).json()
    config.pop("kind")
    config["mode"] = mode
    output, _, _, _ = geometry(shape, config, pooling=True)
    if source.shape != ("time", "batch", math.prod(shape)):
        raise ValueError(
            "pool2d requires a flat (time, batch, channels) spatial signal"
        )
    signal = net.operation(
        "pool2d",
        source,
        name=name,
        shape=("time", "batch", math.prod(output)),
        unit=source.unit,
        signal_type=source.signal_type if mode == "max" else "continuous",
        **config,
    )
    net.operations[-1]["spatial_shape"] = list(output)
    return signal
