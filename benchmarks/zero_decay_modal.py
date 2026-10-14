"""Run zero-decay acceptance and controlled/full-MNIST checks on one Modal L4."""

import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
BASELINE = Path(os.environ.get("SNNLAB_BASELINE_DIR", "/baseline"))
app = modal.App("snnlab-zero-decay")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.10.0",
        "torchvision==0.25.0",
        "numpy",
        "scipy",
        "matplotlib",
        "scikit-learn",
        "h5py",
        "pytest",
        "snntorch==0.9.4",
        "brian2>=2.10",
    )
    .add_local_dir(ROOT / "src", "/candidate/src", ignore=["**/__pycache__/**"])
    .add_local_dir(ROOT / "tests", "/candidate/tests", ignore=["**/__pycache__/**"])
    .add_local_dir(
        ROOT / "benchmarks", "/candidate/benchmarks", ignore=["**/__pycache__/**"]
    )
    .add_local_dir(BASELINE / "src", "/baseline/src", ignore=["**/__pycache__/**"])
)


@app.function(image=image, gpu="L4", cpu=4, memory=16384, timeout=1200)
def validate(only_tests: bool = False):
    import subprocess
    import sys

    import numpy as np

    def run(command, source, capture=True, cpu_only=False):
        env = {
            **os.environ,
            "PYTHONPATH": source + ":/candidate",
            "OMP_NUM_THREADS": "4",
            "MPLBACKEND": "Agg",
        }
        if cpu_only:
            env["CUDA_VISIBLE_DEVICES"] = ""
        completed = subprocess.run(
            command,
            cwd="/candidate",
            env=env,
            text=True,
            capture_output=capture,
            timeout=1100,
        )
        if capture:
            print(completed.stdout, flush=True)
        if completed.returncode:
            if capture:
                print(completed.stderr, flush=True)
            raise RuntimeError(f"Validation failed: {command}")
        return completed.stdout

    tests = run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/sim/test_zero_decay.py",
            "tests/sim/test_gpu_efficiency.py",
            "tests/sim/test_streaming_inference.py",
        ],
        "/candidate/src",
    )
    cpu_tests = run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/sim/test_execution_runtime_state.py",
            "tests/sim/test_execution_training.py",
            "tests/sim/test_execution_checkpoints.py",
        ],
        "/candidate/src",
        cpu_only=True,
    )
    if only_tests:
        return json.dumps({"tests": tests, "cpu_tests": cpu_tests})
    run(
        [sys.executable, "benchmarks/zero_decay.py", "--output", "/tmp/candidate.json"],
        "/candidate/src",
        capture=False,
    )
    run(
        [
            sys.executable,
            "benchmarks/zero_decay.py",
            "--baseline",
            "--reference-weights",
            "/tmp/candidate.weights.npz",
            "--output",
            "/tmp/baseline.json",
        ],
        "/baseline/src",
        capture=False,
    )
    with (
        np.load("/tmp/baseline.agreement.npz") as reference,
        np.load("/tmp/candidate.agreement.npz") as actual,
    ):
        for key in reference.files:
            np.testing.assert_allclose(
                actual[key], reference[key], atol=1e-5, rtol=1e-4, err_msg=key
            )
    return json.dumps(
        {
            "tests": tests,
            "cpu_tests": cpu_tests,
            "baseline_ref": "8a601c7a77047542c9bc4a18e2b45677429beec4",
            "baseline_candidate_agreement": True,
            "baseline": json.loads(Path("/tmp/baseline.json").read_text()),
            "candidate": json.loads(Path("/tmp/candidate.json").read_text()),
        }
    )


@app.local_entrypoint()
def main(output: str = "benchmarks/zero-decay/results.json", only_tests: bool = False):
    result = validate.remote(only_tests)
    Path(output).write_text(json.dumps(json.loads(result), indent=2) + "\n")
    print(f"Saved validated results to {output}")
