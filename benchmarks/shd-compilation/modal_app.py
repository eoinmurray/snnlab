"""Isolated H100 runs; persist source hashes, profiles and measured samples."""

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/workspace")
HERE = ROOT / "benchmarks/shd-compilation"
app = modal.App("snnlab-shd-compilation")
image = (
    (
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
        )
        .env(
            {
                "PYTHONPATH": "/workspace/src:/workspace/benchmarks/shd-compilation",
                "OMP_NUM_THREADS": "4",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            }
        )
        .add_local_dir(ROOT / "src", "/workspace/src", ignore=["**/__pycache__/**"])
        .add_local_dir(ROOT / "tests", "/workspace/tests", ignore=["**/__pycache__/**"])
        .add_local_dir(
            HERE,
            "/workspace/benchmarks/shd-compilation",
            ignore=["results/**", "**/__pycache__/**"],
        )
    )
    if modal.is_local()
    else None
)


@app.function(
    image=image, gpu="H100", cpu=4, memory=32768, timeout=1800, max_containers=1
)
def worker(kind, validate_only=False):
    import io
    import subprocess
    import sys
    import zipfile

    import benchmark as b

    if kind == "coba" or validate_only:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/sim/test_configurable_surrogate.py",
                "tests/sim/test_gpu_efficiency.py",
                "-q",
                "--tb=short",
            ],
            cwd="/workspace",
            text=True,
            capture_output=True,
        )
        print(result.stdout, result.stderr, flush=True)
        if result.returncode:
            raise RuntimeError("GPU regression tests failed")
    if validate_only:
        return json.dumps({"passed": True, "tests": result.stdout}).encode()
    output = Path("/tmp/results")
    report = b.run(kind, output)
    try:
        report["validation"] = b.validate(kind)
    except Exception as error:
        report["validation"] = {"passed": False, "error": str(error)}
    (output / f"{kind}.json").write_text(json.dumps(report, indent=2))
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in output.iterdir():
            handle.write(path, path.name)
    return archive.getvalue()


@app.local_entrypoint()
def main(kind: str = "both", validate_only: bool = False):
    import io
    import zipfile

    destination = HERE / "results"
    destination.mkdir(exist_ok=True)
    if validate_only:
        result = worker.remote("coba", True)
        (destination / "final-cuda-validation.json").write_bytes(result)
        print(result.decode())
        return
    for selected in ("coba", "ping") if kind == "both" else (kind,):
        data = worker.remote(selected)
        with zipfile.ZipFile(io.BytesIO(data)) as handle:
            handle.extractall(destination)
        print(f"Saved {selected} measurements to {destination}")
        report = json.loads((destination / f"{selected}.json").read_text())
        if not report["validation"]["passed"]:
            raise RuntimeError("Numerical validation failed; measurements retained")
