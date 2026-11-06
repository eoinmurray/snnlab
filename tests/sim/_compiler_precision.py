"""Retained recurrent precision gate from the former execution benchmark."""

from contextlib import nullcontext

import torch

from snnlab import lang
from snnlab.sim import GraphExecutor, plan_graph


def circuit(channels, cells, current):
    net = lang.Network("execution-benchmark", dt=1 * lang.ms)
    source = net.input(
        "events", shape=("time", "batch", channels), signal_type="spikes", unit="spike"
    )
    population = net.population(
        "cells",
        size=cells,
        neuron=lang.CUBA_LIF() if current else lang.COBA_LIF(tau_mem=20 * lang.ms),
    )
    synapse = (
        lang.ExponentialCurrent(tau=3 * lang.ms)
        if current
        else lang.AMPA(tau=3 * lang.ms)
    )
    net.connect(
        source,
        population.excitatory,
        name="drive",
        synapse=synapse,
        weight=lang.Constant(4 if current else 0.15),
    )
    net.connect(
        population.spikes,
        population.excitatory,
        name="recurrent",
        connection="recurrent",
        delay=2 * lang.ms,
        synapse=synapse,
        weight=lang.Constant(0.1),
        allow_autapses=False,
    )
    scores = lang.MeanVoltage(source=population.spikes, classes=2, name="readout")
    net.output("scores", scores)
    graph = lang.compile(net).graph
    return graph, tuple(row["id"] for row in graph["parameters"])


def adam_options(execution):
    return {"default": {}, "foreach": {"foreach": True}, "fused": {"fused": True}}[
        execution
    ]


def autocast_context(policy):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if policy == "bfloat16"
        else nullcontext()
    )


def numerical_gate(
    graph,
    trainable,
    device,
    block_steps,
    dtype,
    steps,
    batch,
    adam_execution="default",
    autocast="off",
):
    """Use identical inputs and weights, independent of backend RNG streams."""
    generator = torch.Generator().manual_seed(71)
    inputs = (
        torch.rand(steps, batch, graph["inputs"][0]["shape"][-1], generator=generator)
        < 0.2
    ).to(device=device, dtype=dtype)
    tolerance = 3e-4 if dtype == torch.float32 and autocast == "off" else 5e-2
    results = []
    selected_precision = torch.get_float32_matmul_precision()
    try:
        for size, mode in ((1, None), (block_steps, "default")):
            torch.set_float32_matmul_precision(
                "highest" if mode is None else selected_precision
            )
            model = GraphExecutor(
                plan_graph(graph),
                seed=7,
                trainable_parameters=trainable,
                compile_mode=mode,
                compile_block_steps=size,
            ).to(device=device, dtype=dtype)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=0.001,
                **adam_options("default" if mode is None else adam_execution),
            )
            with autocast_context("off" if mode is None else autocast):
                result, _ = model._forward_impl({"events": inputs}, diagnostics=False)
                loss = result.outputs["scores"].sum()
            loss.backward()
            gradients = {
                name: parameter.grad.detach().clone()
                for name, parameter in model.parameter_map().items()
                if parameter.grad is not None
            }
            optimizer.step()
            model.enforce_constraints()
            results.append((result, gradients, optimizer))
    finally:
        torch.set_float32_matmul_precision(selected_precision)
    expected, actual = results
    assert actual[1].keys() == expected[1].keys(), "gradient availability changed"
    for name, value in expected[0].outputs.items():
        torch.testing.assert_close(
            actual[0].outputs[name], value, atol=tolerance, rtol=tolerance
        )
    for name, value in expected[1].items():
        torch.testing.assert_close(
            actual[1][name], value, atol=tolerance, rtol=tolerance
        )
    for name, value in expected[0].model.parameter_map().items():
        torch.testing.assert_close(
            actual[0].model.parameter_map()[name], value, atol=tolerance, rtol=tolerance
        )
    for name, reference_parameter in expected[0].model.parameter_map().items():
        reference = expected[2].state[reference_parameter]
        candidate = actual[2].state[actual[0].model.parameter_map()[name]]
        assert candidate.keys() == reference.keys(), (
            "optimizer state availability changed"
        )
        for name, value in reference.items():
            torch.testing.assert_close(
                candidate[name],
                value.to(candidate[name].device),
                atol=tolerance,
                rtol=tolerance,
            )
    return {
        "passed": True,
        "atol": tolerance,
        "rtol": tolerance,
        "steps": steps,
        "batch": batch,
        "reference_matmul_precision": "highest",
        "candidate_matmul_precision": selected_precision,
        "adam_execution": adam_execution,
        "reference_adam_execution": "default",
        "reference_autocast": "off",
        "candidate_autocast": autocast,
        "compiler_precision_policy": actual[0].metrics.get("compiler_precision_policy"),
        "compiler_fallback": actual[0].metrics.get("compiler_fallback"),
        "optimizer_state_comparison": "Values compared on the candidate state device; fused Adam keeps its step counter on CUDA while default Adam may keep it on CPU.",
        "checks": ["outputs", "gradients", "Adam parameters", "Adam moments"],
    }
