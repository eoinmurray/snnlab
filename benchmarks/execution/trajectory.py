"""Matched CUDA trajectories and split continuation for one-second presentations."""

import argparse
import json
import time
from pathlib import Path

import torch
from benchmark import circuit, source_provenance, synchronize

from snnlab.sim import GraphExecutor, plan_graph


@torch.no_grad()
def validate(steps, batch, block_steps, coba, device_name="cuda"):
    device = torch.device(device_name)
    graph, trainable = circuit(700, 128, not coba)
    plan = plan_graph(graph)
    signals = tuple(
        [
            f"{population['id']}.{port}"
            for population in plan.populations
            for port in ("voltage", "spikes")
        ]
        + [
            f"{row['id']}.{'current' if row['synapse']['kind'] == 'exponential_current' else 'conductance'}"
            for row in graph["projections"]
        ]
    )
    values = (
        torch.rand(steps, batch, 700, generator=torch.Generator().manual_seed(71)) < 0.2
    ).to(device)
    # Bool spike inputs are deliberately converted to the model's float32 dtype.
    values = values.float()
    groups = (
        "voltages",
        "refractory",
        "conductances",
        "currents",
        "population_histories",
        "input_histories",
        "custom_state",
    )

    def execute(model, inputs, state=None):
        return model._forward(
            {"events": inputs},
            diagnostics=False,
            runtime_state=state,
            required_signals=signals,
        )

    def compare_state(reference, candidate):
        assert reference.completed_steps == candidate.completed_steps
        for group in groups:
            expected, actual = getattr(reference, group), getattr(candidate, group)
            assert expected.keys() == actual.keys()
            for name in expected:
                torch.testing.assert_close(
                    actual[name], expected[name], atol=3e-4, rtol=3e-4
                )

    def compare_samples(reference, candidate):
        for name in signals:
            exact = name.endswith(".spikes")
            torch.testing.assert_close(
                candidate[name],
                reference[name],
                atol=0 if exact else 3e-4,
                rtol=0 if exact else 3e-4,
            )

    synchronize(device)
    started = time.perf_counter()
    models = [
        GraphExecutor(
            plan,
            seed=7,
            trainable_parameters=trainable,
            compile_mode=mode,
            compile_block_steps=size,
        ).to(device)
        for size, mode in ((1, None), (block_steps, "default"))
    ]
    eager, eager_signals = execute(models[0], values)
    compiled, compiled_signals = execute(models[1], values)
    assert compiled.metrics["execution_path"]["selected"] == "compiled_blocks"
    compare_samples(eager_signals, compiled_signals)
    compare_state(eager.runtime_state, compiled.runtime_state)
    for name in eager.outputs:
        torch.testing.assert_close(
            compiled.outputs[name], eager.outputs[name], atol=3e-4, rtol=3e-4
        )
    split = steps * 7 // 16 + 1
    first, first_signals = execute(models[1], values[:split])
    tail, tail_signals = execute(models[1], values[split:], first.runtime_state)
    assert first.metrics["execution_path"]["selected"] == "compiled_blocks"
    assert tail.metrics["execution_path"]["selected"] == "compiled_blocks"
    compare_samples(
        compiled_signals,
        {
            name: torch.cat((first_signals[name], tail_signals[name]))
            for name in signals
        },
    )
    compare_state(compiled.runtime_state, tail.runtime_state)
    synchronize(device)
    return {
        "schema": "snnlab.cuda-trajectory-gate/v1",
        "status": "complete",
        "passed": True,
        "shape": [steps, batch, 700],
        "cells": 128,
        "block_steps": block_steps,
        "coba": coba,
        "split_step": split,
        "signals": signals,
        "state_groups": groups,
        "spikes_exact": True,
        "atol": 3e-4,
        "rtol": 3e-4,
        "elapsed_including_compilation_s": time.perf_counter() - started,
        "execution_path": compiled.metrics["execution_path"],
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "hardware": torch.cuda.get_device_name()
        if device.type == "cuda"
        else str(device),
        "threads": torch.get_num_threads(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "provenance": source_provenance(),
        "limits": [
            "Synthetic float32 trajectories with fixed weights; gradient and Adam equivalence are covered by separate training gates.",
            "No online reductions, callbacks, interventions or recording in this compiled-block gate; those use compatibility paths with separate regression coverage.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--block-steps", type=int, default=16)
    parser.add_argument("--coba", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 4 or args.batch < 1 or not 2 <= args.block_steps <= 64:
        parser.error("steps >=4, batch >=1 and block steps in [2,64] required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = validate(
            args.steps, args.batch, args.block_steps, args.coba, args.device
        )
    except Exception as error:
        args.output.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "provenance": source_provenance(),
                },
                indent=2,
            )
            + "\n"
        )
        raise
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "passed",
                    "shape",
                    "split_step",
                    "elapsed_including_compilation_s",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
