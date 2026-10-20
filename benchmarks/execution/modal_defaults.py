"""Matched Modal benchmark reruns with durable per-policy logs and reports."""

import hashlib
import json
import subprocess
import uuid
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/workspace")
app = modal.App("snnlab-automatic-defaults")
volume = modal.Volume.from_name("snnlab-automatic-defaults", create_if_missing=True)
shd_volume = modal.Volume.from_name("snnlab-shd", create_if_missing=True)
mnist_volume = modal.Volume.from_name(
    "snnlab-snntorch-mnist-comparison", create_if_missing=True
)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.10.0",
        "torchvision==0.25.0",
        "snntorch==1.0.0",
        "numpy>=2.2.6",
        "scipy>=1.15.3",
        "matplotlib>=3.10.8",
        "scikit-learn>=1.7.2",
        "h5py>=3.16.0",
        "pytest>=9",
    )
    .env(
        {
            "PYTHONPATH": "/workspace/src:/workspace",
            "OMP_NUM_THREADS": "4",
            "MPLBACKEND": "Agg",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "SNNLAB_BENCHMARK_COMMIT": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        }
    )
    .add_local_dir(ROOT / "src", "/workspace/src", ignore=["**/__pycache__/**"])
    .add_local_dir(
        ROOT / "benchmarks",
        "/workspace/benchmarks",
        ignore=["**/results/**", "**/artifacts/**", "**/__pycache__/**"],
    )
    .add_local_dir(ROOT / "tests", "/workspace/tests", ignore=["**/__pycache__/**"])
    .add_local_dir(
        ROOT / "examples/customisation",
        "/workspace/examples/customisation",
        ignore=["**/__pycache__/**", "**/network.bundle/**", "**/*.png", "metrics.json"],
    )
    .add_local_file(ROOT / "pyproject.toml", "/workspace/pyproject.toml")
    if modal.is_local()
    else None
)


def execute(run_id, benchmark, smoke, tests=False, device=None):
    import os
    import sys
    import time
    import traceback

    import torch

    os.chdir(ROOT)
    output = Path("/data") / run_id / benchmark
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_id": run_id,
        "benchmark": benchmark,
        "smoke": smoke,
        "device": device,
        "status": "running",
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "torch": str(torch.__version__),
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "src").rglob("*.py"))
        },
        "customisation_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "examples/customisation").glob("*.py"))
        },
        "conditions": {},
    }

    def save():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        volume.commit()

    save()
    try:
        if tests:
            command = [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/sim/test_automatic_speed.py",
                "tests/sim/test_training_replay.py",
                "tests/sim/test_performance.py",
                "tests/sim/test_customisation_example.py",
            ]
            with (output / "tests.log").open("w") as log:
                status = subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, timeout=1200
                )
            manifest["tests_exit_code"] = status.returncode
            save()
            if status.returncode:
                raise RuntimeError("CUDA default policy tests failed")
        policies = (
            ("before", "after", "snntorch")
            if benchmark == "snntorch-comparison" and not tests
            else ("before", "after")
        )
        for policy in policies:
            started = time.time()
            command = [
                sys.executable,
                "benchmarks/execution/defaults.py",
                "--benchmark",
                benchmark,
                "--policy",
                policy,
                "--output",
                str(output / policy),
            ]
            if smoke:
                command.append("--smoke")
            if device is not None:
                command.extend(("--device", device))
            with (output / f"{policy}.log").open("w") as log:
                status = subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, timeout=7200
                )
            manifest["conditions"][policy] = {
                "exit_code": status.returncode,
                "process_wall_s": time.time() - started,
            }
            save()
            if status.returncode:
                raise RuntimeError(f"{policy} failed; inspect durable log")
        manifest["status"] = "complete"
    except Exception:
        manifest.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        save()
    return manifest


resources = dict(
    image=image,
    cpu=4,
    memory=32768,
    timeout=18000,
    volumes={"/data": volume, "/shd": shd_volume, "/mnist": mnist_volume},
)


@app.function(**resources)
def digits(run_id, smoke=False):
    return execute(run_id, "scikit-digits", smoke)


@app.function(gpu="L4", **resources)
def digits_gpu(run_id, smoke=False):
    return execute(run_id, "scikit-digits", smoke, device="cuda:0")


@app.function(gpu="L4", **resources)
def mnist(run_id, smoke=False):
    return execute(run_id, "snntorch-comparison", smoke)


@app.function(gpu="H100", **resources)
def shd(run_id, smoke=False):
    return execute(run_id, "shd", smoke)


@app.function(gpu="H100", **resources)
def validate(run_id):
    return execute(run_id, "snntorch-comparison", True, tests=True)


@app.local_entrypoint()
def main(suite: str = "validate", run_id: str = ""):
    run_id = run_id or "defaults-" + uuid.uuid4().hex[:12]
    names = (
        ("digits", "mnist", "shd")
        if suite == "full"
        else (("validate", "shd") if suite == "smoke" else (suite,))
    )
    jobs = {}
    for name in names:
        function = {
            "digits": digits,
            "digits-gpu": digits_gpu,
            "mnist": mnist,
            "shd": shd,
            "validate": validate,
        }[name]
        job = (
            function.spawn(run_id)
            if name == "validate"
            else function.spawn(run_id, suite == "smoke")
        )
        jobs[name] = job.object_id
    ticket = {"run_id": run_id, "volume": "snnlab-automatic-defaults", "jobs": jobs}
    destination = ROOT / "benchmarks/execution/results/automatic-defaults" / run_id
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "ticket.json").write_text(json.dumps(ticket, indent=2) + "\n")
    print(json.dumps(ticket), flush=True)


def collect(run_id):
    """Collect durable text reports and logs; leave datasets/checkpoints on Modal."""
    import gzip
    import re

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id):
        raise ValueError("invalid run ID")
    destination = ROOT / "benchmarks/execution/results/automatic-defaults" / run_id
    reports = {}
    for benchmark in ("scikit-digits", "snntorch-comparison", "shd"):
        remote = f"/{run_id}/{benchmark}"
        try:
            manifest = json.loads(b"".join(volume.read_file(remote + "/manifest.json")))
        except FileNotFoundError:
            continue
        folder = destination / benchmark
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        reports[benchmark] = {
            "status": manifest["status"],
            "conditions": manifest["conditions"],
        }
        names = ["tests.log"]
        for policy in ("before", "after", "snntorch"):
            if policy not in manifest["conditions"]:
                continue
            names.append(policy + ".log")
            names.extend(
                f"{policy}/{name}"
                for name in (
                    "run.json",
                    "results.json",
                    "summary.json",
                    "metrics.json",
                    "timing.json",
                    "configuration.json",
                )
            )
        for name in names:
            try:
                data = b"".join(volume.read_file(remote + "/" + name))
            except FileNotFoundError:
                continue
            path = folder / (name + ".gz")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(gzip.compress(data, mtime=0))
    print(json.dumps(reports, indent=2))
    return destination
