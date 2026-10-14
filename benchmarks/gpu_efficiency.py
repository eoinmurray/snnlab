"""Synchronized CUDA workload measurements and comparison payloads.

Run with the chosen source checkout on PYTHONPATH. Baseline mode supports 0.10.1.
Timings include separate cold/warm operations; profiler runs are never timed.
"""

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from snnlab import lang
from snnlab.lang import training as recipe
from snnlab.sim.execution import (
    DatasetEncoder,
    DatasetSnapshotBinding,
    DenseArrayBinding,
    ExecutionSpec,
    GraphExecutor,
    RecordingSpec,
    SignalRecording,
    plan_graph,
    simulate,
    train,
)


def make_bundle(width=256):
    net = lang.Network("gpu_efficiency", dt=1 * lang.ms)
    events = net.input(
        "events", shape=("time", "batch", 784), signal_type="spikes", unit="spike"
    )
    cells = net.population("hidden", size=width, neuron=lang.CUBA_LIF())
    net.connect(
        events,
        cells.excitatory,
        name="drive",
        synapse=lang.ExponentialCurrent(tau=3 * lang.ms),
        weight=lang.Normal(mean=2, std=0.2),
    )
    score = lang.readouts.SpikeCount(source=cells.spikes, classes=10, name="scores")
    net.output("class_scores", score)
    training = lang.TrainSpec(
        objectives=[recipe.CrossEntropy(prediction=score, target="label")],
        parameter_groups=[
            recipe.ParameterGroup(
                [p["id"] for p in net.parameters], name="weights", lr=0.001
            )
        ],
        optimizer=recipe.AdamW(weight_decay=0),
        presentation_duration=32 * lang.ms,
    )
    return lang.compile(net, training=training)


def measure(operation, repetitions, *, profile=False):
    torch.cuda.synchronize()
    started = time.perf_counter()
    operation()
    torch.cuda.synchronize()
    cold = time.perf_counter() - started
    for _ in range(2):
        operation()
    torch.cuda.synchronize()
    times = []
    resident = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(repetitions):
        torch.cuda.synchronize()
        start = time.perf_counter()
        operation()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    result = {
        "cold_ms": cold * 1000,
        "median_ms": statistics.median(times),
        "samples_ms": times,
        "resident_allocated_bytes": resident,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_increment_bytes": torch.cuda.max_memory_allocated() - resident,
    }
    if profile:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler:
            operation()
            torch.cuda.synchronize()
        keys = {
            "aten::_local_scalar_dense",
            "aten::clone",
            "aten::nonzero",
            "aten::_to_copy",
            "cudaStreamSynchronize",
        }
        result["operator_counts"] = {
            row.key: row.count for row in profiler.key_averages() if row.key in keys
        }
        result["cuda_activity_count"] = sum(
            1
            for event in profiler.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=7)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    bundle = make_bundle()
    torch.manual_seed(11)
    inputs = (torch.rand(32, 128, 784, device="cuda") < 0.1).float()
    targets = torch.arange(128, device="cuda") % 10
    rows, payloads = {}, {}
    for mode in (None,) if args.baseline else (None, "default"):
        name = "eager" if mode is None else "compiled"
        options = {} if args.baseline else {"compile_mode": mode}
        model = GraphExecutor(
            plan_graph(bundle.graph),
            seed=7,
            trainable_parameters=bundle.training["resolved_parameters"]["trainable"],
            **options,
        ).to("cuda")

        def forward():
            with torch.no_grad():
                return model({"events": inputs}, diagnostics=False).outputs[
                    "class_scores"
                ]

        def backward():
            model.zero_grad(set_to_none=True)
            scores = model({"events": inputs}, diagnostics=False).outputs[
                "class_scores"
            ]
            torch.nn.functional.cross_entropy(scores, targets).backward()
            return scores

        rows[f"{name}_inference"] = measure(forward, args.repetitions, profile=True)
        rows[f"{name}_forward_backward"] = measure(
            backward, args.repetitions, profile=True
        )
        payloads[f"{name}_scores"] = backward().detach().cpu().numpy()
        for parameter, value in model.parameter_map().items():
            payloads[f"{name}_gradient_{parameter}"] = value.grad.detach().cpu().numpy()
        if mode is not None:
            np.testing.assert_allclose(
                payloads["compiled_scores"],
                payloads["eager_scores"],
                atol=2e-5,
                rtol=2e-5,
            )
            for parameter in model.parameter_map():
                np.testing.assert_allclose(
                    payloads[f"compiled_gradient_{parameter}"],
                    payloads[f"eager_gradient_{parameter}"],
                    atol=2e-5,
                    rtol=2e-5,
                )

    # Additional shapes expose dispatch-dominated cases and shape warmup costs.
    for width, batch in ((64, 8), (256, 32)):
        shape_bundle = make_bundle(width)
        shape_inputs = inputs[:, :batch]
        shape_targets = targets[:batch]
        shape_reference = None
        for mode in (None,) if args.baseline else (None, "default"):
            name = "eager" if mode is None else "compiled"
            options = {} if args.baseline else {"compile_mode": mode}
            shape_model = GraphExecutor(
                plan_graph(shape_bundle.graph),
                seed=7,
                trainable_parameters=shape_bundle.training["resolved_parameters"][
                    "trainable"
                ],
                **options,
            ).to("cuda")

            def shape_forward():
                with torch.no_grad():
                    return shape_model(
                        {"events": shape_inputs}, diagnostics=False
                    ).outputs["class_scores"]

            def shape_backward():
                shape_model.zero_grad(set_to_none=True)
                scores = shape_model(
                    {"events": shape_inputs}, diagnostics=False
                ).outputs["class_scores"]
                torch.nn.functional.cross_entropy(scores, shape_targets).backward()
                return scores

            prefix = f"hidden_{width}_batch_{batch}_{name}"
            rows[prefix + "_inference"] = measure(shape_forward, args.repetitions)
            rows[prefix + "_forward_backward"] = measure(
                shape_backward, args.repetitions
            )
            scores = shape_backward().detach().cpu().numpy()
            gradients = [
                p.grad.detach().cpu().numpy() for p in shape_model.parameters()
            ]
            payloads[prefix + "_scores"] = scores
            if shape_reference is None:
                shape_reference = (scores, gradients)
            else:
                np.testing.assert_allclose(
                    scores, shape_reference[0], atol=2e-5, rtol=2e-5
                )
                for actual, expected in zip(gradients, shape_reference[1]):
                    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)

    spec = ExecutionSpec(
        kind="train",
        graph=bundle.graph,
        training=bundle.training,
        input_bindings=(DenseArrayBinding("events", inputs),),
        targets={"label": targets},
        device="cuda",
        diagnostics=False,
        updates=8,
        seed=7,
    )
    rows["training_8_updates"] = measure(
        lambda: train(spec), args.repetitions, profile=True
    )
    trained = train(spec)
    for parameter, value in trained.parameters.items():
        payloads[f"trained_{parameter}"] = value.cpu().numpy()
    payloads["training_losses"] = np.asarray(
        [row["loss"] for row in trained.metrics["updates"]]
    )

    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "dataset.npz"
        np.savez(
            path,
            features=np.random.default_rng(5).random((1024, 784), dtype=np.float32),
            labels=np.arange(1024) % 10,
        )
        binding = DatasetSnapshotBinding(
            path=path,
            input_id="events",
            dataset_id="gpu-efficiency",
            split="test",
            encoder=DatasetEncoder(
                "rate_poisson", duration_ms=32, max_rate_hz=100, seed=17
            ),
        )
        for depth in (0,) if args.baseline else (0, 2):
            options = {} if args.baseline else {"dataset_prefetch": depth}
            dataset_spec = ExecutionSpec(
                kind="infer",
                graph=bundle.graph,
                input_bindings=(binding,),
                device="cuda",
                diagnostics=False,
                batch_size=128,
                seed=7,
                **options,
            )
            rows[f"dataset_prefetch_{depth}"] = measure(
                lambda: simulate(dataset_spec), args.repetitions, profile=True
            )
            payloads[f"dataset_scores_{depth}"] = (
                simulate(dataset_spec).outputs["class_scores"].cpu().numpy()
            )
        if not args.baseline:
            np.testing.assert_array_equal(
                payloads["dataset_scores_0"], payloads["dataset_scores_2"]
            )

    record_model = GraphExecutor(plan_graph(bundle.graph), seed=7).to("cuda")
    for block_steps in (1,) if args.baseline else (1, 32):
        options = {} if args.baseline else {"block_steps": block_steps}
        recording = RecordingSpec(
            (
                SignalRecording("hidden.spikes", kind="spike_events"),
                SignalRecording("hidden.voltage"),
            ),
            sink=lambda block: None,
            sink_id="benchmark/discard/v1",
            **options,
        )

        def record():
            with torch.no_grad():
                return record_model(
                    {"events": inputs}, diagnostics=False, recording=recording
                )

        rows[f"recording_block_{block_steps}"] = measure(
            record, args.repetitions, profile=True
        )

    if not args.baseline:
        timing_spec = ExecutionSpec(
            kind="infer",
            graph=bundle.graph,
            input_bindings=(DenseArrayBinding("events", inputs),),
            device="cuda",
            diagnostics=False,
        )
        timing = simulate(timing_spec)
        torch.cuda.synchronize()
        start = time.perf_counter()
        timing = simulate(timing_spec)
        torch.cuda.synchronize()
        rows["timing_check"] = {
            "reported_simulate_ms": timing.metrics["simulate_s"] * 1000,
            "external_total_ms": (time.perf_counter() - start) * 1000,
            "scope": timing.metrics["timing_scope"],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output.with_suffix(".npz"), **payloads)
    args.output.write_text(
        json.dumps(
            {
                "torch": str(torch.__version__),
                "gpu": torch.cuda.get_device_name(),
                "baseline": args.baseline,
                "workload": {
                    "input": 784,
                    "hidden": 256,
                    "classes": 10,
                    "steps": 32,
                    "batch": 128,
                    "dtype": "float32",
                    "tf32": False,
                    "repetitions": args.repetitions,
                    "warmups": 2,
                },
                "results": rows,
            },
            indent=2,
        )
    )
    print(args.output.read_text())


if __name__ == "__main__":
    main()
