"""Spatial geometry for flat, channel-major network signals."""

from __future__ import annotations

import math
from collections.abc import Mapping


def dimensions(value, *, name="spatial_shape", rank=3, minimum=1):
    if not isinstance(value, (tuple, list)) or len(value) != rank:
        raise ValueError(f"{name} must contain {rank} integers")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < minimum for v in value):
        raise ValueError(f"{name} must contain integers >= {minimum}")
    return tuple(value)


def pair(value, *, name, minimum=1):
    if isinstance(value, int) and not isinstance(value, bool):
        value = (value, value)
    return dimensions(value, name=name, rank=2, minimum=minimum)


def geometry(shape, config, *, pooling=False):
    channels, height, width = dimensions(shape)
    kernel = pair(config["kernel_size"], name="kernel_size")
    stride = pair(config.get("stride", 1), name="stride")
    padding = pair(config.get("padding", 0), name="padding", minimum=0)
    if pooling and any(p > k // 2 for p, k in zip(padding, kernel, strict=True)):
        raise ValueError("pooling padding must not exceed half the kernel size")
    output = tuple(
        (n + 2 * p - k) // s + 1
        for n, p, k, s in zip((height, width), padding, kernel, stride, strict=True)
    )
    if min(output) < 1:
        raise ValueError("spatial kernel produces an empty output")
    return (channels, *output), kernel, stride, padding


def signal_row(graph: Mapping, signal: str):
    owner, _, port = signal.partition(".")
    for collection in ("inputs", "populations", "operations"):
        for row in graph.get(collection, []):
            if row["id"] == owner:
                if collection != "populations" and port != "value":
                    raise ValueError(f"invalid spatial signal: {signal}")
                return row
    raise ValueError(f"unresolved spatial signal: {signal}")


def signal_spatial_shape(graph, signal):
    row = signal_row(graph, signal)
    shape = dimensions(row.get("spatial_shape"))
    flat = row.get("size", row.get("shape", [None])[-1])
    if flat != math.prod(shape):
        raise ValueError(f"{signal}: spatial_shape does not match flat signal size")
    return shape


def source_root(graph, signal, *, projection=True):
    """Follow instantaneous spatial operations to their population or input source."""
    operations = {f"{row['id']}.value": row for row in graph.get("operations", [])}
    seen = set()
    while signal in operations:
        if signal in seen:
            raise ValueError("spatial operation sources form a cycle")
        seen.add(signal)
        row = operations[signal]
        if row["kind"] != "pool2d" or len(row["sources"]) != 1:
            if not projection:
                break
            raise ValueError(
                "projection operation sources must be spatial pooling chains"
            )
        signal = row["sources"][0]
    return signal


def projection_shape(graph, row):
    config = row.get("connectivity")
    if config is None:
        target = next(
            p
            for p in graph["populations"]
            if p["id"] == row["target"].partition(".")[0]
        )
        source = signal_row(graph, row["source"])
        size = source.get("size", source.get("shape", [None])[-1])
        return [target["size"], size]
    if config.get("kind") != "conv2d":
        raise ValueError("unsupported projection connectivity")
    source_shape = signal_spatial_shape(graph, row["source"])
    target_shape = signal_spatial_shape(
        graph, row["target"].partition(".")[0] + ".spikes"
    )
    output, kernel, _, _ = geometry(source_shape, config)
    if target_shape[1:] != output[1:]:
        raise ValueError(
            f"{row['id']}: convolution output does not match target spatial_shape"
        )
    if source_root(graph, row["source"]).partition(".")[0] == row["target"].partition(
        "."
    )[0] and not row.get("allow_autapses", True):
        raise ValueError("same-population convolution requires allow_autapses=True")
    return [target_shape[0], source_shape[0] * math.prod(kernel)]


def pool_tensor(value, shape, config):
    """Pool arbitrary leading axes, retaining the public flat channel axis."""
    import torch.nn.functional as F

    _, kernel, stride, padding = geometry(shape, config, pooling=True)
    leading = value.shape[:-1]
    spatial = value.reshape(-1, *shape)
    if config["mode"] == "avg":
        result = F.avg_pool2d(spatial, kernel, stride, padding, count_include_pad=False)
    else:
        result = F.max_pool2d(spatial, kernel, stride, padding)
    return result.reshape(*leading, -1)


def convolution_drive(source, weight, shape, config):
    import torch.nn.functional as F

    _, kernel, stride, padding = geometry(shape, config)
    # Public parameter matrices remain [fan_in, out_channels] at runtime.
    filters = weight.T.reshape(weight.shape[1], shape[0], *kernel)
    result = F.conv2d(
        source.reshape(-1, *shape), filters, stride=stride, padding=padding
    )
    return result.flatten(1)


def _validate_spatial_shapes(graph):
    for collection in ("inputs", "populations", "operations"):
        for row in graph.get(collection, []):
            if "spatial_shape" in row:
                shape = dimensions(row["spatial_shape"])
                flat = row.get("size", row.get("shape", [None])[-1])
                if flat != math.prod(shape):
                    raise ValueError(
                        f"{row['id']}: spatial_shape does not match flat size"
                    )
                if collection != "populations" and row.get("shape") != [
                    "time",
                    "batch",
                    flat,
                ]:
                    raise ValueError(
                        "spatial signals require flat (time, batch, channels) shape"
                    )


def validate_spatial(graph):
    _validate_spatial_shapes(graph=graph)

    for row in graph.get("operations", []):
        if row["kind"] != "pool2d":
            continue
        source_root(graph, f"{row['id']}.value", projection=False)
        if len(row["sources"]) != 1 or row.get("parameters"):
            raise ValueError("pool2d requires one source and no parameters")
        shape = signal_spatial_shape(graph, row["sources"][0])
        config = row.get("config", {})
        if config.get("mode") not in {"avg", "max"}:
            raise ValueError("pooling mode must be 'avg' or 'max'")
        output, _, _, _ = geometry(shape, config, pooling=True)
        if dimensions(row.get("spatial_shape")) != output:
            raise ValueError(
                "pooling output spatial_shape does not match inferred geometry"
            )
        source = signal_row(graph, row["sources"][0])
        port = row["sources"][0].partition(".")[2]
        source_type = source.get(
            "signal_type",
            "spikes"
            if port == "spikes"
            else "voltage"
            if port in {"voltage", "pre_reset_voltage"}
            else "continuous",
        )
        expected_type = source_type if config["mode"] == "max" else "continuous"
        if row.get("signal_type") != expected_type:
            raise ValueError("pooling signal_type does not match its mode and source")
    parameters = {p["id"]: p for p in graph.get("parameters", [])}
    for row in graph.get("projections", []):
        if row["source"].partition(".")[0] in {
            op["id"] for op in graph.get("operations", [])
        }:
            source_root(graph, row["source"])
        if "connectivity" in row:
            expected = projection_shape(graph, row)
            for parameter in row.get("parameters", []):
                if (
                    parameter in parameters
                    and parameters[parameter]["shape"] != expected
                ):
                    raise ValueError(
                        f"{row['id']}: convolution kernel shape must be {expected}"
                    )
