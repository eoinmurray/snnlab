"""Experimental fixed-shape training capture; never enabled by the simulator.

Inputs must already be CUDA tensors. Each replay starts a fresh presentation,
performs backward, capturable Adam and declared built-in constraints. Changing
shape requires a fresh capture. Native callbacks, recording and continuation
are outside this experiment and retain their normal validation paths.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from benchmark import circuit, source_provenance

from snnlab.sim import GraphExecutor, plan_graph


def experiment(steps, batch, channels, cells, block_steps, repeats, coba):
    device = torch.device("cuda:0")
    graph, trainable = circuit(channels, cells, not coba)
    generator = torch.Generator(device=device).manual_seed(71)
    inputs = [
        (
            torch.rand(steps, batch, channels, device=device, generator=generator) < 0.2
        ).float()
        for _ in range(3)
    ]
    target = torch.arange(batch, device=device) % 2

    def setup():
        model = GraphExecutor(
            plan_graph(graph),
            seed=7,
            trainable_parameters=trainable,
            compile_mode="default",
            compile_block_steps=block_steps,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, capturable=True)
        return model, optimizer

    def step(model, optimizer, value):
        optimizer.zero_grad(set_to_none=False)
        result, _ = model._forward(
            {"events": value},
            diagnostics=False,
            retain_parameters=False,
            retain_final_state=False,
            retain_runtime_state=False,
        )
        torch.nn.functional.cross_entropy(result.outputs["scores"], target).backward()
        optimizer.step()
        model.enforce_constraints()
        return result

    model, optimizer = setup()
    initial = {name: p.detach().clone() for name, p in model.parameter_map().items()}

    def reset(current, opt):
        with torch.no_grad():
            for name, parameter in current.parameter_map().items():
                parameter.copy_(initial[name])
                if parameter.grad is not None:
                    parameter.grad.zero_()
            for state in opt.state.values():
                for value in state.values():
                    value.zero_()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    static = inputs[0].clone()
    with torch.cuda.stream(stream):
        for _ in range(3):
            step(model, optimizer, static)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    warmup_s = time.perf_counter() - started
    reset(model, optimizer)
    torch.cuda.synchronize()
    captured = torch.cuda.CUDAGraph()
    started = time.perf_counter()
    with torch.cuda.graph(captured):
        captured_result = step(model, optimizer, static)
    torch.cuda.synchronize()
    capture_s = time.perf_counter() - started
    reset(model, optimizer)
    reference, reference_optimizer = setup()
    # The capture itself performs one update; reset all optimizer state and
    # weights before comparing repeated fresh-presentation updates.
    checks = []
    for index, value in enumerate(inputs):
        static.copy_(value)
        captured.replay()
        expected = step(reference, reference_optimizer, value)
        for name, output in expected.outputs.items():
            torch.testing.assert_close(
                captured_result.outputs[name], output, atol=3e-4, rtol=3e-4
            )
        for name, parameter in reference.parameter_map().items():
            actual = model.parameter_map()[name]
            assert (actual.grad is None) == (parameter.grad is None)
            if parameter.grad is not None:
                torch.testing.assert_close(
                    actual.grad, parameter.grad, atol=3e-4, rtol=3e-4
                )
            torch.testing.assert_close(actual, parameter, atol=3e-4, rtol=3e-4)
            for key, expected_state in reference_optimizer.state[parameter].items():
                torch.testing.assert_close(
                    optimizer.state[actual][key], expected_state, atol=3e-4, rtol=3e-4
                )
        checks.append({"update": index + 1, "passed": True})

    def measure(replay):
        reset(
            model if replay else reference, optimizer if replay else reference_optimizer
        )
        samples = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            started = time.perf_counter()
            for value in inputs:
                if replay:
                    static.copy_(value)
                    captured.replay()
                else:
                    step(reference, reference_optimizer, value)
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - started)
        return samples

    loop = measure(False)
    replay = measure(True)
    return {
        "schema": "snnlab.cuda-graph-experiment/v1",
        "status": "complete",
        "hardware": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "provenance": source_provenance(),
        "graph": graph,
        "shape": [steps, batch, channels],
        "cells": cells,
        "block_steps": block_steps,
        "dtype": "float32",
        "coba": coba,
        "warmup_s": warmup_s,
        "capture_s": capture_s,
        "checks": checks,
        "checks_scope": [
            "outputs",
            "gradients",
            "parameters",
            "Adam moments/step",
            "changed inputs",
            "fresh presentation state",
        ],
        "atol": 3e-4,
        "rtol": 3e-4,
        "loop_three_updates_s": loop,
        "replay_three_updates_s": replay,
        "loop_samples_per_s": 3 * batch / statistics.median(loop),
        "replay_samples_per_s": 3 * batch / statistics.median(replay),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "execution_path": captured_result.metrics["execution_path"],
        "limits": [
            "Experimental benchmark only; no public replay mode or default change.",
            "Fixed shape, built-in model, no recording/callbacks/continuation; shape changes need recapture.",
            "Timing starts from resident inputs and includes device input copy, fresh presentation, forward/backward, capturable Adam and constraints; encoding and host transfers excluded.",
            "Numerical reference uses the same compiled blocks and capturable optimizer; separate execution benchmark gates compare these blocks to eager.",
            "Synthetic execution does not establish task accuracy.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--channels", type=int, default=700)
    parser.add_argument("--cells", type=int, default=128)
    parser.add_argument("--block-steps", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--coba", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        min(args.steps, args.batch, args.channels, args.cells, args.repeats) < 1
        or not 1 <= args.block_steps <= 64
    ):
        parser.error("sizes must be positive and block steps in [1, 64]")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = experiment(
            args.steps,
            args.batch,
            args.channels,
            args.cells,
            args.block_steps,
            args.repeats,
            args.coba,
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
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "shape",
                    "warmup_s",
                    "capture_s",
                    "loop_samples_per_s",
                    "replay_samples_per_s",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
