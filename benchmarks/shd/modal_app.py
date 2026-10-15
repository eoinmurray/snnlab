"""Submit detached SHD H100 jobs; collect persisted reports independently later."""

import argparse
import json
import re
import uuid
from pathlib import Path

import modal

LOCAL_ROOT = (
    Path(__file__).resolve().parents[2] if modal.is_local() else Path("/workspace")
)
app = modal.App("snnlab-shd")
VOLUME = "snnlab-shd"
DATA_ROOT = Path("/data")
volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
image = None
if modal.is_local():
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "torch==2.10.0",
            "torchvision==0.25.0",
            "numpy>=2.2.6",
            "scipy>=1.15.3",
            "matplotlib>=3.10.8",
            "scikit-learn>=1.7.2",
            "h5py>=3.16.0",
        )
        .env(
            {
                "PYTHONPATH": "/workspace/src:/workspace/benchmarks/shd",
                "OMP_NUM_THREADS": "4",
            }
        )
        .add_local_dir(
            LOCAL_ROOT / "src", "/workspace/src", ignore=["**/__pycache__/**"]
        )
        .add_local_dir(
            LOCAL_ROOT / "benchmarks/shd",
            "/workspace/benchmarks/shd",
            ignore=["results/**", "**/__pycache__/**"],
        )
    )


@app.function(
    image=image,
    gpu="H100",
    cpu=4,
    memory=16384,
    timeout=7200,
    volumes={"/data": volume},
    max_containers=1,
)
def worker(
    run_id: str,
    epochs: int,
    smoke: bool,
    trace_mode: str = "off",
    trace_skip_updates: int = 1,
    trace_updates: int = 3,
    trace_detailed_labels: bool = True,
):
    from dataclasses import replace

    from benchmark import run
    from shd_recipe import Config

    validate_run_id(run_id)
    validate_options(epochs, trace_mode, trace_skip_updates, trace_updates)
    cfg = Config(epochs=epochs)
    if smoke:
        cfg = replace(cfg, epochs=2, steps=64)
    volume.reload()
    output = DATA_ROOT / "runs" / run_id
    try:
        run(
            cfg,
            output,
            DATA_ROOT / "cache",
            device="cuda:0",
            smoke=smoke,
            trace_mode=trace_mode,
            trace_skip_updates=trace_skip_updates,
            trace_updates=trace_updates,
            trace_detailed_labels=trace_detailed_labels,
        )
        names = ["configuration.json", "metrics.json", "timing.json", "summary.json"]
        if trace_mode != "off":
            names.append("diagnostic_metrics.json")
            names.extend(
                str(path.relative_to(output))
                for path in (output / "tracing").iterdir()
                if path.name != "trace.json"
            )
        cloud = cloud_paths(run_id)
        cloud.update(
            status="complete",
            report_files=names,
            trace=f"runs/{run_id}/tracing/trace.json"
            if (output / "tracing/trace.json").exists()
            else None,
        )
        # This manifest and all results persist without any local result handler.
        (output / "cloud.json").write_text(json.dumps(cloud, indent=2) + "\n")
        return {"run_id": run_id, "status": "complete"}
    finally:
        volume.commit()


def validate_run_id(run_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id):
        raise ValueError(
            "run ID must contain 1–80 letters, digits, underscores or hyphens"
        )
    return run_id


def cloud_paths(run_id):
    validate_run_id(run_id)
    return {
        "volume": VOLUME,
        "run_id": run_id,
        "directory": f"runs/{run_id}",
        "checkpoint": f"runs/{run_id}/final.checkpoint",
        "snapshot": f"runs/{run_id}/spikes.npz",
    }


def validate_options(epochs, trace_mode, trace_skip_updates, trace_updates):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if trace_mode not in {"off", "monitor", "profile", "both"}:
        raise ValueError("trace mode must be off, monitor, profile or both")
    if trace_skip_updates < 0 or trace_updates < 1:
        raise ValueError("invalid trace update window")


def submit(
    epochs=10,
    smoke=False,
    trace_mode="off",
    trace_skip_updates=1,
    trace_updates=3,
    trace_detailed_labels=True,
):
    """Submit inside a detached App context, then return without waiting."""
    validate_options(epochs, trace_mode, trace_skip_updates, trace_updates)
    run_id = ("smoke-" if smoke else "shd-") + uuid.uuid4().hex[:12]
    print(f"Submitting {run_id}", flush=True)
    call = worker.spawn(
        run_id,
        epochs,
        smoke,
        trace_mode,
        trace_skip_updates,
        trace_updates,
        trace_detailed_labels,
    )
    ticket = {**cloud_paths(run_id), "call_id": call.object_id, "status": "submitted"}
    destination = LOCAL_ROOT / "benchmarks/shd/results" / run_id
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "cloud.json").write_text(json.dumps(ticket, indent=2) + "\n")
    print(
        f"Submitted {run_id}; call={call.object_id}; ticket={destination / 'cloud.json'}",
        flush=True,
    )
    print(
        f"Collect later: python benchmarks/shd/modal_app.py --mode collect --run-id {run_id}",
        flush=True,
    )
    return ticket


def collect(run_id):
    """Read persisted reports directly from the Volume; no worker invocation."""
    cloud = cloud_paths(run_id)
    remote = f"/{cloud['directory']}"
    # A completion manifest is written only after all requested work succeeds.
    # Missing manifests propagate rather than collecting an unfinished run.
    try:
        completed = json.loads(b"".join(volume.read_file(f"{remote}/cloud.json")))
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"No completion manifest for {run_id}; check the Modal dashboard and collect after the run succeeds"
        ) from exc
    if completed.get("status") != "complete" or completed.get("run_id") != run_id:
        raise ValueError("remote manifest does not describe this completed run")
    names = completed["report_files"]
    for name in names:
        path = Path(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.suffix not in {".json", ".csv"}
        ):
            raise ValueError("invalid report path in remote manifest")
    destination = LOCAL_ROOT / "benchmarks/shd/results" / run_id
    destination.mkdir(parents=True, exist_ok=True)
    ticket_path = destination / "cloud.json"
    ticket = json.loads(ticket_path.read_text()) if ticket_path.exists() else {}
    for name in names:
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"".join(volume.read_file(f"{remote}/{name}")))
    ticket_path.write_text(json.dumps({**ticket, **completed}, indent=2) + "\n")
    print(f"Collected reports: {destination}")
    return destination


@app.local_entrypoint()
def main(
    epochs: int = 10,
    smoke: bool = False,
    trace_mode: str = "off",
    trace_skip_updates: int = 1,
    trace_updates: int = 3,
    trace_detailed_labels: bool = True,
    mode: str = "launch",
    run_id: str = "",
):
    """Modal CLI entrypoint: use modal run --detach for launch mode."""
    if mode == "collect":
        return collect(run_id)
    if mode != "launch":
        raise ValueError("mode must be launch or collect")
    return submit(
        epochs,
        smoke,
        trace_mode,
        trace_skip_updates,
        trace_updates,
        trace_detailed_labels,
    )


def cli():
    """Direct Python CLI always launches with detach=True; collection is read-only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("launch", "collect"), default="launch")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--trace-mode", choices=("off", "monitor", "profile", "both"), default="off"
    )
    parser.add_argument("--trace-skip-updates", type=int, default=1)
    parser.add_argument("--trace-updates", type=int, default=3)
    parser.add_argument(
        "--trace-detailed-labels", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if args.mode == "collect":
        return collect(args.run_id)
    validate_options(
        args.epochs, args.trace_mode, args.trace_skip_updates, args.trace_updates
    )
    with modal.enable_output(), app.run(detach=True):
        return submit(
            args.epochs,
            args.smoke,
            args.trace_mode,
            args.trace_skip_updates,
            args.trace_updates,
            args.trace_detailed_labels,
        )


if __name__ == "__main__":
    cli()
