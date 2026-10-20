"""Bounded CUDA capture for deterministic native forward/loss/backward work.

Optimizer, constraints, observations and persistence stay in the normal trainer.
Warmup never updates weights and restores RNG streams and gradient bindings.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import torch

from snnlab import extensions as E
from snnlab.sim.graph.validation import deferred_validation, validate_pending
from snnlab.sim.training.observations import evaluation_rng


@dataclass
class _Capture:
    graph: object
    inputs: dict
    targets: dict
    gradients: tuple
    result: tuple
    validation: list


class TrainingReplay:
    """Capture at most two repeated shapes per train call; report every fallback."""

    def __init__(self, spec, model, training, scheduled, *, tracing=False):
        self.policy = spec.cuda_graphs
        self.parameters = tuple(model.parameters())
        self.uses = Counter(len(indices) for _, _, indices in scheduled)
        self.entries = {}
        self.failed = {}
        self.stream = None
        self.replays = 0
        self.normal_updates = 0
        self.fallbacks = Counter()
        self.reasons = []
        if self.policy == "off":
            self.reasons.append("disabled by request")
        if not self.parameters or self.parameters[0].device.type != "cuda":
            self.reasons.append("CUDA device required")
        if spec.compile_mode is None:
            self.reasons.append("compilation disabled")
        if tracing:
            self.reasons.append("execution tracing")
        if spec.runtime_state is not None:
            self.reasons.append("runtime continuation")
        for data in (model.plan.graph, training):
            for row in E.required(data):
                category = row["category"]
                if category in {
                    "neuron",
                    "synapse",
                    "surrogate",
                    "operation",
                    "objective",
                    "regularizer",
                    "constraint",
                }:
                    if not E.get(category, row["definition"]).capture_safe:
                        self.reasons.append(
                            f"extension requires capture_safe=True: {row['definition']}"
                        )
        if model._check_extension_values and any(
            E.required(data) for data in (model.plan.graph, training)
        ):
            self.reasons.append("strict extension value validation")
        if any(
            row.get("kind", "").startswith("custom_") and not row.get("definition")
            for row in (
                *training.get("objectives", ()),
                *training.get("regularizers", ()),
            )
        ):
            self.reasons.append("custom objective or regularizer")
        if any(parameter.dtype != torch.float32 for parameter in self.parameters):
            self.reasons.append("float32 parameters required")

    @staticmethod
    def _key(inputs, targets):
        return tuple(
            (group, name, tuple(value.shape), value.dtype, value.device, value.stride())
            for group, values in (("input", inputs), ("target", targets))
            for name, value in sorted(values.items())
        )

    def _fallback(self, reasons):
        self.normal_updates += 1
        self.fallbacks.update(reasons)
        return None

    def run(self, inputs, targets, step):
        """Return the captured numerical result, or None for normal execution."""
        if self.reasons:
            return self._fallback(self.reasons)
        if not torch.is_grad_enabled():
            return self._fallback(["autograd disabled"])
        values = (*inputs.values(), *targets.values())
        if any(value.requires_grad for value in values):
            return self._fallback(["input or target gradients"])
        if any(value.device.type != "cuda" for value in values):
            return self._fallback(["inputs and targets must be CUDA tensors"])
        batch = next(iter(inputs.values())).shape[1]
        if self.uses[batch] < 2:
            return self._fallback(["shape occurs only once"])
        key = self._key(inputs, targets)
        if key in self.failed:
            return self._fallback([self.failed[key]])
        if key not in self.entries:
            if len(self.entries) >= 2:
                return self._fallback(["capture cache limit (2 shapes)"])
            entry = self._capture(inputs, targets, step)
            if entry is None:
                return self._fallback([self.failed[key]])
            self.entries[key] = entry
        entry = self.entries[key]
        for name, value in inputs.items():
            entry.inputs[name].copy_(value)
        for name, value in targets.items():
            entry.targets[name].copy_(value)
        for parameter, gradient in zip(self.parameters, entry.gradients):
            parameter.grad = gradient
        entry.graph.replay()
        validate_pending(entry.validation)
        self.replays += 1
        return entry.result

    def _capture(self, inputs, targets, step):
        key = self._key(inputs, targets)
        previous = tuple(parameter.grad for parameter in self.parameters)
        graph = None
        captured = False
        try:
            static_inputs = {name: value.clone() for name, value in inputs.items()}
            static_targets = {name: value.clone() for name, value in targets.items()}
            device = next(iter(inputs.values())).device
            if self.stream is None:
                self.stream = torch.cuda.Stream(device=device)
            stream = self.stream
            stream.wait_stream(torch.cuda.current_stream(device))
            for parameter in self.parameters:
                parameter.grad = None

            def execute():
                for parameter in self.parameters:
                    if parameter.grad is not None:
                        parameter.grad.zero_()
                with deferred_validation() as checks:
                    result = step(static_inputs, static_targets)
                return result, checks

            # Built-in capture candidates are deterministic. Isolation also
            # protects preparation/evaluation streams from compiler warmup.
            with evaluation_rng():
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        _, checks = execute()
                        validate_pending(checks)
                torch.cuda.current_stream(device).wait_stream(stream)
                torch.cuda.synchronize(device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    result, checks = execute()
                torch.cuda.current_stream(device).wait_stream(stream)
            captured = True
            return _Capture(
                graph,
                static_inputs,
                static_targets,
                tuple(parameter.grad for parameter in self.parameters),
                result,
                checks,
            )
        except torch.cuda.OutOfMemoryError:
            self.failed[key] = "CUDA capture memory unavailable"
        except RuntimeError as error:
            message = str(error).lower()
            if not any(
                text in message
                for text in (
                    "operation not permitted when stream is capturing",
                    "operation failed due to a previous error during capture",
                    "cudastreamcapture",
                    "cuda graph capture",
                )
            ):
                raise
            self.failed[key] = "operation unsupported during CUDA capture"
        finally:
            if not captured:
                if self.stream is not None:
                    self.stream.synchronize()
                for parameter, gradient in zip(self.parameters, previous):
                    parameter.grad = gradient
                if graph is not None:
                    graph.reset()
        return None

    def summary(self):
        return {
            "schema": "snnlab.training-cuda-graphs/v1",
            "policy": self.policy,
            "selected": "cuda_graphs" if self.replays else "normal",
            "scope": "forward, loss and backward; optimizer, clipping, constraints and reporting stay outside capture",
            "captures": len(self.entries),
            "replayed_updates": self.replays,
            "normal_updates": self.normal_updates,
            "fallback_reasons": dict(self.fallbacks),
            "cache_limit": 2,
        }
