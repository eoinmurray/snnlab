"""Command-line graph simulation and training."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from snnlab.sim.bundle import load_graph_bundle, load_training_recipe
from snnlab.sim.execution import (
    DatasetEncoder,
    DatasetSnapshotBinding,
    PoissonInputBinding,
    execute_request,
    execution_spec_from_args,
    load_dense_array_bindings,
    load_event_stream_bindings,
    load_runtime_state,
    load_target_array_bindings,
    save_runtime_state,
    write_inference_artifacts,
)
from snnlab.sim.timing import duration_steps


def _add_common_arguments(parser):
    parser.add_argument(
        "--bundle",
        type=str,
        default=None,
        help="Load an authenticated snnlab.lang graph bundle. "
        "Training additionally requires an authenticated training.json.",
    )
    parser.add_argument(
        "--executor",
        choices=("graph",),
        default="graph",
        help="Graph execution backend.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Graph execution device: auto, cpu, cuda, cuda:N, or mps (default: auto).",
    )
    parser.add_argument(
        "--diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Return exposed graph diagnostics (default: enabled; use --no-diagnostics to disable).",
    )
    parser.add_argument(
        "--load-runtime-state",
        type=str,
        default=None,
        help="Restore complete graph-executor dynamic state from a runtime-state directory.",
    )
    parser.add_argument(
        "--save-runtime-state",
        type=str,
        default=None,
        help="Save complete graph-executor dynamic state to a runtime-state directory.",
    )
    parser.add_argument(
        "--t-ms",
        type=float,
        default=200.0,
        help="Total simulation duration in ms (default: 200). "
        "Metrics are measured over the full trace; notebooks strip any "
        "startup transient in post.",
    )
    parser.add_argument(
        "--input-rate",
        type=float,
        default=25.0,
        dest="spike_rate",
        help="Baseline input rate in Hz (default: 25)",
    )
    parser.add_argument(
        "--input-rates",
        type=float,
        nargs="+",
        default=None,
        help="Categorical Poisson rates in Hz. One rate is "
        "sampled uniformly and independently per image presentation.",
    )
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Graph initialization and input-encoding seed (default: 0).",
    )


def _add_sim_arguments(parser):
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit a dataset snapshot to N samples.",
    )
    parser.add_argument(
        "--load-weights",
        type=str,
        default=None,
        help="Graph checkpoint directory or PyTorch state dictionary for inference.",
    )
    parser.add_argument(
        "--scale-projection",
        action="append",
        default=[],
        metavar="ID=FACTOR",
        help="[graph] Multiply a named graph projection for this inference request; repeatable.",
    )
    parser.add_argument(
        "--measurement",
        metavar="JSON",
        default=None,
        help="[graph] Absolute start_step/end_step reduction window as JSON.",
    )
    parser.add_argument(
        "--recording",
        metavar="JSON",
        default=None,
        help="[graph] Signal/cell selections and optional NPZ sink directory as JSON.",
    )
    parser.add_argument(
        "--retain-outputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="[graph] Retain named output tensors; sinks receive outputs either way.",
    )
    parser.add_argument(
        "--reset-voltage",
        action="append",
        default=[],
        metavar="JSON",
        help="[graph] Selective leaky-integrator voltage reset as JSON; repeatable.",
    )
    parser.add_argument(
        "--decisions",
        default=None,
        metavar="JSON",
        help="[graph] Decision segment boundaries and spike counts as JSON.",
    )
    parser.add_argument(
        "--intervention",
        action="append",
        default=[],
        metavar="KIND:POPULATION=VALUE",
        help="[graph] Ordered spike intervention: drop:POPULATION=PROBABILITY, add:POPULATION=RATE_HZ or replay:POPULATION=PATH@sha256:FILE_DIGEST; repeatable.",
    )
    parser.add_argument(
        "--inference-timestep-ms",
        type=float,
        default=None,
        help="[graph Poisson] Recompile an immutable graph copy at this inference timestep.",
    )
    parser.add_argument(
        "--n-batch",
        type=int,
        default=64,
        help="Number of generated Poisson trials (default: 64).",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Dense NPY/NPZ replay, bound by graph input id.",
    )
    parser.add_argument(
        "--event-file",
        type=str,
        default=None,
        help="Sparse event-stream NPZ replay for graph execution. Coordinates are "
        "zero-based integer steps, batches, and channels.",
    )
    parser.add_argument(
        "--dataset-file",
        type=str,
        default=None,
        help="Immutable NPZ dataset snapshot for graph execution.",
    )
    parser.add_argument(
        "--dataset-encoder",
        choices=("rate-poisson", "prebinned-spikes", "event-bin"),
        default=None,
        help="Standard encoder applied to --dataset-file.",
    )
    parser.add_argument("--dataset-input-id", type=str, default=None)
    parser.add_argument("--dataset-target-id", type=str, default=None)
    parser.add_argument("--dataset-feature-key", default="features")
    parser.add_argument("--dataset-label-key", default="labels")
    parser.add_argument(
        "--poisson-protocol",
        choices=["fixed-rate", "categorical-rate"],
        default=None,
        help="Generate graph-input Poisson spikes from --input-rate or --input-rates.",
    )
    parser.add_argument(
        "--input-dataset-id",
        type=str,
        default=None,
        help="Stable dataset or snapshot identity recorded for a graph replay.",
    )
    parser.add_argument(
        "--input-split",
        type=str,
        default=None,
        help="Dataset split recorded for a graph replay.",
    )
    parser.add_argument(
        "--input-shuffle",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Record whether the graph replay batch was shuffled (use --no-input-shuffle for false).",
    )


def _add_train_arguments(parser):
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Dense NPY/NPZ graph-training inputs, bound by graph input id.",
    )
    parser.add_argument(
        "--target-file",
        type=str,
        default=None,
        help="Integer NPY/NPZ graph-training targets, bound by recipe target id.",
    )
    parser.add_argument(
        "--dataset-file",
        type=str,
        default=None,
        help="Immutable NPZ dataset snapshot for graph training.",
    )
    parser.add_argument(
        "--dataset-encoder",
        choices=("rate-poisson", "prebinned-spikes", "event-bin"),
        default=None,
    )
    parser.add_argument("--dataset-input-id", type=str, default=None)
    parser.add_argument("--dataset-target-id", type=str, default=None)
    parser.add_argument("--dataset-feature-key", default="features")
    parser.add_argument("--dataset-label-key", default="labels")
    parser.add_argument(
        "--input-dataset-id",
        type=str,
        default=None,
        help="Stable dataset or snapshot identity recorded for graph training.",
    )
    parser.add_argument(
        "--input-split",
        type=str,
        default=None,
        help="Dataset split recorded for graph training.",
    )
    parser.add_argument(
        "--input-shuffle",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Deterministically shuffle graph-training samples each epoch.",
    )
    parser.add_argument(
        "--load-weights",
        type=str,
        default=None,
        help="Resume graph training from a training-checkpoint directory.",
    )
    parser.add_argument(
        "--save-final-checkpoint",
        type=str,
        default=None,
        help="Write the final graph-training checkpoint directory.",
    )
    parser.add_argument(
        "--checkpoint-selection",
        type=str,
        default=None,
        help="Graph checkpoint selection policy as JSON (split, cadence, metric, tie_break, include_initial).",
    )
    parser.add_argument(
        "--save-selected-checkpoint",
        type=str,
        default=None,
        help="Write the selected graph-training checkpoint directory.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs; defaults to the bundle recipe.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Mini-batch size for DataLoader. Default: 64 (from models.BATCH_SIZE).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit dataset to N samples (smoke test)",
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="snnsim",
        allow_abbrev=False,
        description="Simulate and train snnlab.lang graph bundles.",
    )
    parent = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    _add_common_arguments(parent)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    simulation = subparsers.add_parser(
        "sim",
        parents=[parent],
        allow_abbrev=False,
        help="Simulate a graph, optionally loading learned parameters.",
    )
    _add_sim_arguments(simulation)
    training = subparsers.add_parser(
        "train",
        parents=[parent],
        allow_abbrev=False,
        help="Train a graph using its authenticated training recipe.",
    )
    _add_train_arguments(training)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.bundle is None:
        raise SystemExit("graph execution requires an explicit --bundle")
    if args.out_dir is None:
        raise SystemExit(
            f"{args.mode} writes run artifacts and requires an explicit --out-dir"
        )
    try:
        request = execution_spec_from_args(args)
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
    input_file = getattr(args, "input_file", None)
    event_file = getattr(args, "event_file", None)
    poisson_protocol = getattr(args, "poisson_protocol", None)
    dataset_file = getattr(args, "dataset_file", None)
    if (
        sum(
            bool(value)
            for value in (input_file, event_file, poisson_protocol, dataset_file)
        )
        != 1
    ):
        raise SystemExit(
            "graph execution requires exactly one of --input-file, --event-file, --poisson-protocol, or --dataset-file"
        )

    try:
        manifest, graph = load_graph_bundle(args.bundle)
    except (ValueError, TypeError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
    try:
        inference_overrides = {}
        projection_scales = {}
        for item in getattr(args, "scale_projection", []):
            projection_id, separator, raw_factor = item.partition("=")
            if not separator or not projection_id or not raw_factor:
                raise ValueError("--scale-projection expects ID=FACTOR")
            if projection_id in projection_scales:
                raise ValueError(
                    f"--scale-projection repeats projection {projection_id!r}"
                )
            projection_scales[projection_id] = float(raw_factor)
        if projection_scales:
            inference_overrides["projection_scales"] = projection_scales
        if getattr(args, "inference_timestep_ms", None) is not None:
            inference_overrides["timestep_ms"] = args.inference_timestep_ms
        if dataset_file:
            if not args.dataset_encoder:
                raise ValueError("--dataset-file requires --dataset-encoder")
            if not args.input_dataset_id or not args.input_split:
                raise ValueError(
                    "--dataset-file requires --input-dataset-id and --input-split"
                )
            input_ids = [row["id"] for row in graph.get("inputs", [])]
            input_id = args.dataset_input_id or (
                input_ids[0] if len(input_ids) == 1 else None
            )
            if input_id is None:
                raise ValueError(
                    "multi-input dataset graphs require --dataset-input-id"
                )
            encoder_kind = args.dataset_encoder.replace("-", "_")
            encoder = DatasetEncoder(
                encoder_kind,
                duration_ms=(
                    args.t_ms if encoder_kind in {"rate_poisson", "event_bin"} else None
                ),
                max_rate_hz=(
                    args.spike_rate
                    if encoder_kind == "rate_poisson" and not args.input_rates
                    else None
                ),
                seed=request.seed if encoder_kind == "rate_poisson" else 0,
                rates_hz=(
                    tuple(args.input_rates)
                    if encoder_kind == "rate_poisson" and args.input_rates
                    else None
                ),
            )
            binding_update = {
                "input_bindings": (
                    DatasetSnapshotBinding(
                        path=Path(dataset_file),
                        input_id=input_id,
                        target_id=args.dataset_target_id,
                        dataset_id=args.input_dataset_id,
                        split=args.input_split,
                        encoder=encoder,
                        feature_key=args.dataset_feature_key,
                        label_key=args.dataset_label_key,
                        sample_cap=args.max_samples,
                        shuffle=bool(args.input_shuffle),
                        order_seed=request.seed,
                    ),
                )
            }
        elif poisson_protocol:
            input_ids = [row["id"] for row in graph.get("inputs", [])]
            if len(input_ids) != 1:
                raise ValueError(
                    "CLI Poisson generation requires exactly one graph input"
                )
            dt_ms = float(graph["timebase"]["dt"]["value"])
            steps = duration_steps(args.t_ms, dt_ms)
            rates = (
                args.input_rates
                if poisson_protocol == "categorical-rate"
                else [args.spike_rate]
            )
            if poisson_protocol == "categorical-rate" and not rates:
                raise ValueError("categorical-rate Poisson requires --input-rates")
            binding_update = {
                "input_bindings": (
                    PoissonInputBinding(
                        input_id=input_ids[0],
                        steps_count=int(steps),
                        batch_size=args.n_batch,
                        rates_hz=rates,
                        seed=args.seed,
                        categorical=poisson_protocol == "categorical-rate",
                    ),
                )
            }
        else:
            binding_update = (
                {"input_bindings": load_event_stream_bindings(event_file, graph)}
                if event_file
                else {"input_bindings": load_dense_array_bindings(input_file, graph)}
            )
        target_update = {}
        if request.kind == "train":
            target_file = getattr(args, "target_file", None)
            if not target_file and not dataset_file:
                raise ValueError("graph training requires --target-file")
            if dataset_file and not args.dataset_target_id:
                raise ValueError("graph dataset training requires --dataset-target-id")
            recipe = load_training_recipe(args.bundle, manifest, graph)
            if args.epochs is None:
                request = replace(request, epochs=int(recipe["epochs"]))
            target_update = {
                "training": recipe,
                **(
                    {"target_bindings": load_target_array_bindings(target_file, recipe)}
                    if target_file
                    else {}
                ),
            }
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    runtime_state = (
        load_runtime_state(args.load_runtime_state, device=request.device)
        if getattr(args, "load_runtime_state", None)
        else None
    )
    request = replace(
        request,
        **binding_update,
        **target_update,
        protocol={
            "dataset": {
                key: value
                for key, value in {
                    "identity": getattr(args, "input_dataset_id", None),
                    "split": getattr(args, "input_split", None),
                    "sample_cap": getattr(args, "max_samples", None),
                    "shuffle": getattr(args, "input_shuffle", None),
                }.items()
                if value is not None
            }
        },
        options={
            **request.options,
            **(
                {"inference_overrides": inference_overrides}
                if inference_overrides
                else {}
            ),
        },
        runtime_state=runtime_state,
    )
    result = execute_request(request)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_inference_artifacts(out_dir, result, graph=graph, seed=request.seed)
    if getattr(args, "save_runtime_state", None):
        assert result.runtime_state is not None
        save_runtime_state(args.save_runtime_state, result.runtime_state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
