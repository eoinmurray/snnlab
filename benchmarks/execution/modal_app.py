"""Detached, serial H100 validation with durable reports and independent collection."""

import json
import re
import subprocess
import uuid
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/workspace")
VOLUME = "snnlab-execution"
volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
app = modal.App("snnlab-execution")
image = (
    (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "torch==2.10.0",
            "torchvision==0.25.0",
            "numpy>=2.2.6",
            "scipy>=1.15.3",
            "matplotlib>=3.10.8",
            "scikit-learn>=1.7.2",
            "h5py>=3.16.0",
            "pytest>=9",
        )
        .env(
            {
                "PYTHONPATH": "/workspace/src:/workspace/benchmarks/execution",
                "OMP_NUM_THREADS": "1",
                "MPLBACKEND": "Agg",
            }
        )
        .add_local_dir(ROOT / "src", "/workspace/src", ignore=["**/__pycache__/**"])
        .add_local_dir(
            ROOT / "benchmarks",
            "/workspace/benchmarks",
            ignore=["**/results/**", "**/__pycache__/**"],
        )
        .add_local_dir(ROOT / "tests", "/workspace/tests", ignore=["**/__pycache__/**"])
        .add_local_file(ROOT / "pyproject.toml", "/workspace/pyproject.toml")
    )
    if modal.is_local()
    else None
)


def valid_id(run_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id):
        raise ValueError("invalid run ID")
    return run_id


@app.function(
    image=image,
    gpu="H100",
    cpu=4,
    memory=32768,
    timeout=10800,
    volumes={"/data": volume},
    max_containers=1,
)
def worker(run_id: str, identity: dict, suite: str = "full"):
    import os
    import sys
    import time

    valid_id(run_id)
    volume.reload()
    output = Path("/data/runs") / run_id
    output.mkdir(parents=True, exist_ok=False)
    (ROOT / ".execution-git.json").write_text(json.dumps(identity))
    os.chdir(ROOT)
    benchmark = [sys.executable, "benchmarks/execution/benchmark.py"]
    common = ["--devices", "cuda", "--repeats", "3", "--case-timeout", "240"]
    tests = [
        "tests/sim/" + name + ".py"
        for name in (
            "test_execution_dtypes",
            "test_network_blocks",
            "test_device_preparation",
            "test_dtype_artifacts",
            "test_native_device_callbacks",
            "test_constraint_boundaries",
            "test_gpu_efficiency",
            "test_compiled_extensions",
        )
    ]
    jobs = [
        ("regressions", [sys.executable, "-m", "pytest", "-q", *tests], 900),
        (
            "current-short",
            [
                *benchmark,
                *common,
                "--steps",
                "33",
                "--batch-sizes",
                "8",
                "--block-sizes",
                "1",
                "4",
                "16",
                "32",
                "64",
            ],
            1800,
        ),
        (
            "coba-128",
            [
                *benchmark,
                *common,
                "--coba",
                "--channels",
                "700",
                "--cells",
                "128",
                "--steps",
                "128",
                "--batch-sizes",
                "8",
                "32",
                "--block-sizes",
                "1",
                "4",
                "16",
                "--profile",
            ],
            2400,
        ),
        (
            "coba-1000",
            [
                *benchmark,
                *common,
                "--coba",
                "--channels",
                "700",
                "--cells",
                "128",
                "--steps",
                "1000",
                "--batch-sizes",
                "8",
                "32",
                "256",
                "--block-sizes",
                "1",
                "16",
                "32",
            ],
            3000,
        ),
        (
            "recording-prefetch",
            [
                *benchmark,
                *common,
                "--steps",
                "128",
                "--batch-sizes",
                "8",
                "--block-sizes",
                "1",
                "--mode",
                "simulate",
                "--record",
                "--prefetch",
                "2",
                "--profile",
            ],
            600,
        ),
        (
            "bfloat16",
            [
                *benchmark,
                *common,
                "--steps",
                "128",
                "--batch-sizes",
                "32",
                "--block-sizes",
                "4",
                "16",
                "--dtype",
                "bfloat16",
            ],
            900,
        ),
        (
            "float16",
            [
                *benchmark,
                *common,
                "--steps",
                "128",
                "--batch-sizes",
                "32",
                "--block-sizes",
                "4",
                "--dtype",
                "float16",
            ],
            600,
        ),
    ]
    for steps, batch in ((33, 8), (128, 32), (1000, 32)):
        jobs.append(
            (
                f"cuda-graph-{steps}",
                [
                    sys.executable,
                    "benchmarks/execution/cuda_graph.py",
                    "--coba",
                    "--steps",
                    str(steps),
                    "--batch",
                    str(batch),
                    "--block-steps",
                    "16",
                ],
                900,
            )
        )
    if suite == "graphs":
        jobs = [row for row in jobs if row[0].startswith("cuda-graph-")]
    elif suite == "final-performance":
        replay = next(row for row in jobs if row[0] == "cuda-graph-1000")
        jobs = [
            (
                "coba-final-1000",
                [
                    *benchmark,
                    *common,
                    "--coba",
                    "--channels",
                    "700",
                    "--cells",
                    "128",
                    "--steps",
                    "1000",
                    "--batch-sizes",
                    "256",
                    "--block-sizes",
                    "1",
                    "16",
                ],
                1800,
            ),
            replay,
        ]
    elif suite == "native-replay-benchmark":
        jobs = [
            (
                "native-replay",
                [sys.executable, "benchmarks/execution/native_replay.py"],
                1200,
            ),
        ]
    elif suite in {"native-replay", "native-replay-final"}:
        jobs = [
            (
                "native-replay-regressions",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "tests/sim/test_training_replay.py",
                    "tests/sim/test_execution_api.py",
                    "tests/sim/test_native_device_callbacks.py",
                ],
                1200,
            ),
        ]
        if suite == "native-replay-final":
            jobs.append(
                (
                    "native-replay",
                    [sys.executable, "benchmarks/execution/native_replay.py"],
                    1200,
                )
            )
    elif suite == "compatibility":
        jobs = [
            (
                "compatibility-regressions",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    *tests,
                    "tests/sim/test_compiler_precision.py",
                    "tests/sim/test_execution_api.py",
                ],
                900,
            ),
        ]
    elif suite == "final-precision":
        jobs = [
            (
                "final-regressions",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    *tests,
                    "tests/sim/test_compiler_precision.py",
                ],
                900,
            ),
            (
                "bfloat16-final",
                [
                    *benchmark,
                    *common,
                    "--dtype",
                    "bfloat16",
                    "--steps",
                    "128",
                    "--batch-sizes",
                    "32",
                    "--block-sizes",
                    "1",
                    "4",
                    "16",
                ],
                1200,
            ),
            (
                "float16-final",
                [
                    *benchmark,
                    *common,
                    "--dtype",
                    "float16",
                    "--steps",
                    "128",
                    "--batch-sizes",
                    "32",
                    "--block-sizes",
                    "4",
                ],
                600,
            ),
            (
                "amp-final",
                [
                    *benchmark,
                    *common,
                    "--autocast",
                    "bfloat16",
                    "--coba",
                    "--channels",
                    "700",
                    "--cells",
                    "128",
                    "--steps",
                    "128",
                    "--batch-sizes",
                    "32",
                    "--block-sizes",
                    "16",
                ],
                900,
            ),
        ]
    elif suite == "rounding":
        jobs = [
            (
                "bfloat16-rounding",
                [
                    sys.executable,
                    "benchmarks/execution/precision_probe.py",
                    *common,
                    "--dtype",
                    "bfloat16",
                    "--steps",
                    "128",
                    "--batch-sizes",
                    "32",
                    "--block-sizes",
                    "1",
                    "4",
                    "16",
                ],
                1200,
            )
        ]
    elif suite == "trajectory":
        jobs = [
            (
                "trajectory-current",
                [sys.executable, "benchmarks/execution/trajectory.py"],
                900,
            ),
            (
                "trajectory-coba",
                [sys.executable, "benchmarks/execution/trajectory.py", "--coba"],
                900,
            ),
        ]
    elif suite in {"precision", "fused", "amp"}:
        base = [
            *benchmark,
            *common,
            "--coba",
            "--channels",
            "700",
            "--cells",
            "128",
            "--steps",
            "128",
            "--batch-sizes",
            "32",
            "--block-sizes",
            "16",
        ]
        jobs = [
            ("tf32-high", [*base, "--matmul-precision", "high"], 900),
            ("fused-adam", [*base, "--adam-execution", "fused"], 900),
            ("amp-bfloat16", [*base, "--autocast", "bfloat16"], 900),
        ]
        if suite == "fused":
            jobs = [row for row in jobs if row[0] == "fused-adam"]
        elif suite == "amp":
            jobs = [row for row in jobs if row[0] == "amp-bfloat16"]
    elif suite == "recording":
        jobs = [
            (
                "recording-regressions",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "tests/sim/test_gpu_efficiency.py",
                    "tests/sim/test_dtype_artifacts.py",
                ],
                600,
            ),
            (
                "recording-final",
                [
                    *benchmark,
                    *common,
                    "--steps",
                    "128",
                    "--batch-sizes",
                    "8",
                    "--block-sizes",
                    "1",
                    "--mode",
                    "simulate",
                    "--record",
                    "--prefetch",
                    "2",
                    "--profile",
                ],
                600,
            ),
        ]
    elif suite in {"diagnostics", "profile"}:
        jobs = [
            (
                "boundary-regressions",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "tests/sim/test_constraint_boundaries.py",
                    "tests/sim/test_shd_benchmark.py",
                ],
                600,
            ),
            (
                "cuda-profile",
                [
                    *benchmark,
                    *common,
                    "--coba",
                    "--channels",
                    "700",
                    "--cells",
                    "128",
                    "--steps",
                    "128",
                    "--batch-sizes",
                    "32",
                    "--block-sizes",
                    "1",
                    "16",
                    "--profile",
                ],
                1200,
            ),
            (
                "cuda-recording-profile",
                [
                    *benchmark,
                    *common,
                    "--steps",
                    "128",
                    "--batch-sizes",
                    "8",
                    "--block-sizes",
                    "1",
                    "--mode",
                    "simulate",
                    "--record",
                    "--prefetch",
                    "2",
                    "--profile",
                ],
                600,
            ),
            (
                "cpu-comparison",
                [
                    *benchmark,
                    "--devices",
                    "cpu",
                    "--coba",
                    "--channels",
                    "700",
                    "--cells",
                    "128",
                    "--steps",
                    "128",
                    "1000",
                    "--batch-sizes",
                    "32",
                    "--block-sizes",
                    "1",
                    "16",
                    "--repeats",
                    "3",
                    "--case-timeout",
                    "240",
                ],
                1800,
            ),
            (
                "native-shd",
                [
                    sys.executable,
                    "benchmarks/shd/benchmark.py",
                    "--smoke",
                    "--device",
                    "cuda",
                    "--compile-mode",
                    "default",
                    "--cpu-threads",
                    "1",
                    "--output",
                    str(output / "native-shd"),
                ],
                600,
            ),
        ]
        if suite == "profile":
            jobs = [row for row in jobs if row[0] == "cuda-profile"]
    elif suite != "full":
        raise ValueError(
            "suite must be full, graphs, trajectory, precision, fused, amp, recording, profile or diagnostics"
        )
    manifest = {
        "run_id": run_id,
        "volume": VOLUME,
        "status": "running",
        "suite": suite,
        "jobs": [],
        "report_files": [],
    }
    try:
        for name, command, timeout in jobs:
            if not name.endswith("regressions") and name != "native-shd":
                command += ["--output", str(output / f"{name}.json")]
            started = time.time()
            print(f"Starting {name}", flush=True)
            with (output / f"{name}.log").open("w") as log:
                try:
                    process = subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout
                    )
                    result = {
                        "exit_code": process.returncode,
                        "status": "complete" if process.returncode == 0 else "failed",
                    }
                except subprocess.TimeoutExpired:
                    result = {"status": "timeout", "limit_s": timeout}
            manifest["jobs"].append(
                {
                    "name": name,
                    "command": command,
                    "elapsed_s": time.time() - started,
                    **result,
                }
            )
            manifest["report_files"] = sorted(
                str(path.relative_to(output))
                for path in output.rglob("*")
                if path.is_file()
                and path.suffix in {".json", ".log"}
                and path.name != "cloud.json"
            )
            (output / "cloud.json").write_text(json.dumps(manifest, indent=2) + "\n")
            volume.commit()
            print(f"Finished {name}: {result}", flush=True)
        manifest["status"] = (
            "complete"
            if all(row["status"] == "complete" for row in manifest["jobs"])
            else "completed_with_failures"
        )
        return manifest
    finally:
        (output / "cloud.json").write_text(json.dumps(manifest, indent=2) + "\n")
        volume.commit()


def collect(run_id):
    valid_id(run_id)
    remote = f"/runs/{run_id}"
    manifest = json.loads(b"".join(volume.read_file(f"{remote}/cloud.json")))
    if manifest["run_id"] != run_id:
        raise ValueError("run ID mismatch")
    destination = ROOT / "artifacts/execution" / run_id
    destination.mkdir(parents=True, exist_ok=True)
    for name in manifest["report_files"]:
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or Path(name).suffix not in {".json", ".log"}
        ):
            raise ValueError("invalid report path")
        (destination / name).parent.mkdir(parents=True, exist_ok=True)
        (destination / name).write_bytes(b"".join(volume.read_file(f"{remote}/{name}")))
    (destination / "cloud.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Collected {manifest['status']}: {destination}", flush=True)
    return manifest


@app.local_entrypoint()
def main(mode: str = "launch", run_id: str = "", suite: str = "full"):
    if mode == "collect":
        return collect(run_id)
    if mode != "launch":
        raise ValueError("mode must be launch or collect")
    run_id = "cuda-" + uuid.uuid4().hex[:12]
    identity = {
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)
        ),
        "snapshot_transport": "Modal add_local_dir",
    }
    if suite not in {
        "full",
        "graphs",
        "final-performance",
        "final-precision",
        "compatibility",
        "native-replay",
        "native-replay-final",
        "native-replay-benchmark",
        "trajectory",
        "rounding",
        "precision",
        "fused",
        "amp",
        "recording",
        "profile",
        "diagnostics",
    }:
        raise ValueError(
            "suite must be full, graphs, trajectory, precision, fused, amp, recording, profile or diagnostics"
        )
    call = worker.spawn(run_id, identity, suite)
    ticket = {
        "run_id": run_id,
        "volume": VOLUME,
        "call_id": call.object_id,
        "status": "submitted",
    }
    destination = ROOT / "artifacts/execution" / run_id
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "ticket.json").write_text(json.dumps(ticket, indent=2) + "\n")
    print(json.dumps(ticket), flush=True)
