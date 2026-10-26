"""Bounded forward-only CUDA replay, with independent returned tensor storage."""

from collections import Counter
from dataclasses import fields, is_dataclass, replace

import torch
from torch import nn

from snnlab import extensions as E
from snnlab.sim.diagnostics.tracing import current_trace
from snnlab.sim.graph.validation import deferred_validation, validate_pending
from snnlab.sim.training.observations import evaluation_rng


def _clone(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, nn.Module):
        return value
    if is_dataclass(value):
        return replace(
            value,
            **{
                row.name: _clone(getattr(value, row.name))
                for row in fields(value)
                if row.init
            },
        )
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    if isinstance(value, list):
        return [_clone(item) for item in value]
    return value


def _replay_fallback_reason(model, inputs, kwargs):
    reason = None
    if model.cuda_graphs == "off":
        reason = "disabled by request"
    elif torch.is_grad_enabled():
        reason = "forward autograd enabled"
    elif not inputs or any(
        value.device.type != "cuda" or value.requires_grad for value in inputs.values()
    ):
        reason = "CUDA inputs without input gradients required"
    elif current_trace() is not None:
        reason = "execution tracing"
    elif (
        model.surrogate is not None
        and not E.resolve("surrogate", model.surrogate).capture_safe
    ):
        reason = "surrogate requires capture_safe=True"
    elif model.compile_mode is None:
        reason = "compilation disabled"
    elif any(
        kwargs.get(name)
        for name in (
            "runtime_state",
            "recording",
            "interventions",
            "resets",
            "decisions",
            "measurement",
        )
    ):
        reason = "continuation, recording or intervention controls"
    elif any(
        not E.get(row["category"], row["definition"]).capture_safe
        for row in E.required(model.plan.graph)
        if row["category"]
        in {"neuron", "synapse", "surrogate", "operation", "constraint"}
    ):
        reason = "extension requires capture_safe=True"
    elif model._check_extension_values and E.required(model.plan.graph):
        reason = "strict extension value validation"
    return reason


class ForwardReplay:
    """Capture the third use of a compatible shape; keep at most two graphs."""

    def __init__(self):
        self.uses = Counter()
        self.entries = {}
        self.failed = {}
        self.fallbacks = Counter()
        self.replays = 0

    def run(self, model, inputs, kwargs, step):
        reason = _replay_fallback_reason(model=model, inputs=inputs, kwargs=kwargs)

        if reason:
            self.fallbacks[reason] += 1
            return None
        key = (
            tuple(
                (name, tuple(value.shape), value.dtype, value.device, value.stride())
                for name, value in sorted(inputs.items())
            ),
            tuple(
                (name, repr(value))
                for name, value in sorted(kwargs.items())
                if name != "batch_offset"
            ),
            tuple(
                (value.data_ptr(), value.dtype, value.device)
                for value in model.parameters()
            ),
            model.precision,
            model.compile_block_steps,
            torch.is_autocast_enabled("cuda"),
            torch.get_autocast_dtype("cuda"),
        )
        if key not in self.uses and len(self.uses) >= 16:
            self.fallbacks["shape tracking limit (16 shapes)"] += 1
            return None
        self.uses[key] += 1
        if key in self.failed:
            self.fallbacks[self.failed[key]] += 1
            return None
        if key not in self.entries:
            if self.uses[key] < 3:
                self.fallbacks["shape warmup"] += 1
                return None
            if len(self.entries) >= 2:
                self.fallbacks["capture cache limit (2 shapes)"] += 1
                return None
            static = {name: value.clone() for name, value in inputs.items()}
            stream = torch.cuda.Stream(device=next(iter(inputs.values())).device)
            stream.wait_stream(torch.cuda.current_stream(stream.device))
            graph = None
            try:
                with evaluation_rng(), torch.cuda.stream(stream):
                    step(static)
                torch.cuda.current_stream(stream.device).wait_stream(stream)
                torch.cuda.synchronize(stream.device)
                graph = torch.cuda.CUDAGraph()
                with (
                    torch.cuda.graph(graph, stream=stream),
                    deferred_validation() as checks,
                ):
                    result = step(static)
                torch.cuda.current_stream(stream.device).wait_stream(stream)
                self.entries[key] = (graph, static, result, checks)
            except RuntimeError as error:
                message = str(error).lower()
                if not isinstance(error, torch.cuda.OutOfMemoryError) and not any(
                    text in message for text in ("capture", "capturing")
                ):
                    raise
                stream.synchronize()
                if graph is not None:
                    graph.reset()
                self.failed[key] = (
                    "CUDA capture unavailable: " + str(error).splitlines()[0][:240]
                )
                self.fallbacks[self.failed[key]] += 1
                return None
        graph, static, result, checks = self.entries[key]
        for name, value in inputs.items():
            static[name].copy_(value)
        graph.replay()
        validate_pending(checks)
        self.replays += 1
        copied = _clone(result)
        copied[0].metrics["inference_retention"]["window"]["batch_offset"] = kwargs.get(
            "batch_offset", 0
        )
        return copied

    def summary(self):
        return {
            "scope": "forward only",
            "captures": len(self.entries),
            "replayed_calls": self.replays,
            "fallback_reasons": dict(self.fallbacks),
            "cache_limit": 2,
        }
