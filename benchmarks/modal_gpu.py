"""Run the CUDA acceptance tests and paired benchmarks on a single Modal L4.

Requires the optional Modal CLI. Set SNNLAB_BASELINE_DIR to an archive of the
baseline commit's src/ directory. Only explicit source/test/benchmark directories
are uploaded; local config, credentials, docs edits and datasets are excluded.
"""

import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
BASELINE = Path(os.environ.get("SNNLAB_BASELINE_DIR", "/baseline"))
app = modal.App("snnlab-gpu-efficiency")
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
        "snntorch>=0.9.4,<1.1",
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

    def run(command, source):
        env = {
            **os.environ,
            "PYTHONPATH": source + ":/candidate",
            "OMP_NUM_THREADS": "4",
            "MPLBACKEND": "Agg",
        }
        result = subprocess.run(
            command,
            cwd="/candidate",
            env=env,
            text=True,
            capture_output=True,
            timeout=900,
        )
        print(result.stdout)
        if result.returncode:
            print(result.stderr)
            raise RuntimeError(f"Command failed with {result.returncode}: {command}")
        return result.stdout

    tests = run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/sim/test_gpu_efficiency.py",
            "tests/sim/test_adaptive_neurons.py",
            "tests/sim/test_streaming_inference.py",
            "tests/sim/test_extension_validation.py",
        ],
        "/candidate/src",
    )
    if only_tests:
        return json.dumps({"tests": tests})
    for name, source in (
        ("baseline", "/baseline/src"),
        ("candidate", "/candidate/src"),
    ):
        command = [
            sys.executable,
            "benchmarks/gpu_efficiency.py",
            "--output",
            f"/tmp/{name}.json",
        ]
        if name == "baseline":
            command.append("--baseline")
        run(command, source)
    reference, actual = np.load("/tmp/baseline.npz"), np.load("/tmp/candidate.npz")
    for key in reference.files:
        np.testing.assert_allclose(
            actual[key], reference[key], atol=2e-5, rtol=2e-5, err_msg=key
        )
    return json.dumps(
        {
            "tests": tests,
            "baseline_candidate_agreement": True,
            "baseline": json.loads(Path("/tmp/baseline.json").read_text()),
            "candidate": json.loads(Path("/tmp/candidate.json").read_text()),
        }
    )


@app.local_entrypoint()
def main(
    output: str = "benchmarks/gpu-efficiency-results.json", only_tests: bool = False
):
    result = validate.remote(only_tests)
    Path(output).write_text(json.dumps(json.loads(result), indent=2) + "\n")
    print(f"Saved verified results to {output}")
