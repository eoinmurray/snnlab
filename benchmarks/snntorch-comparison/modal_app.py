"""Run training, profile saved checkpoints, or collect results from Modal."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/workspace")
HERE = ROOT / "benchmarks/snntorch-comparison"
VOLUME = "snnlab-snntorch-mnist-comparison"
app = modal.App("snnlab-snntorch-mnist-comparison")
volume = modal.Volume.from_name(VOLUME, create_if_missing=True)


def snapshot():
    paths = sorted((ROOT / "src").rglob("*.py")) + sorted(HERE.glob("*.py"))
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def source_hash():
    return hashlib.sha256(json.dumps(snapshot(), sort_keys=True).encode()).hexdigest()


image = None
if modal.is_local():
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
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
        )
        .env(
            {
                "PYTHONPATH": "/workspace/src:/workspace/benchmarks/snntorch-comparison",
                "OMP_NUM_THREADS": "4",
                "MPLBACKEND": "Agg",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "SNNLAB_BENCHMARK_COMMIT": commit,
            }
        )
        .add_local_dir(ROOT / "src", "/workspace/src", ignore=["**/__pycache__/**"])
        .add_local_dir(
            HERE,
            "/workspace/benchmarks/snntorch-comparison",
            ignore=["results/**", "**/__pycache__/**"],
        )
    )


@app.function(
    image=image,
    volumes={"/results": volume},
    gpu="L4",
    cpu=4,
    memory=16384,
    timeout=21600,
    max_containers=2,
)
def train_worker(backend: str, run_id: str):
    import benchmark as b
    import torch

    volume.reload()
    root = Path("/results") / run_id
    packet = torch.load(root / "prepared.pt", map_location="cpu", weights_only=True)
    result = b.run_training(
        backend,
        b.Config(**packet["config"]),
        packet,
        root / f"{backend}.pt",
        "cuda:0",
        on_epoch=volume.commit,
    )
    return json.dumps(result)


@app.function(
    image=image, volumes={"/results": volume}, cpu=4, memory=8192, timeout=28800
)
def train_pair(run_id: str, expected_source: str):
    import benchmark as b

    if source_hash() != expected_source:
        raise RuntimeError("uploaded source mismatch")
    root = Path("/results") / run_id
    if (root / "results.json").exists():
        raise RuntimeError("this source snapshot has already completed training")
    config = b.Config()
    b.prepare(config, "/results/mnist", root / "prepared.pt")
    volume.commit()
    jobs = {name: train_worker.spawn(name, run_id) for name in ("snntorch", "snnlab")}
    training = {name: json.loads(job.get()) for name, job in jobs.items()}
    cloud = {
        "volume": VOLUME,
        "prepared": f"{run_id}/prepared.pt",
        "checkpoints": {name: f"{run_id}/{name}.pt" for name in jobs},
        "native_training_checkpoint": f"{run_id}/snnlab.native",
    }
    report = b.combined_results(training, run_id=run_id, cloud=cloud)
    b.write_json(root / "results.json", report)
    volume.commit()
    return json.dumps(report)


@app.function(
    image=image,
    volumes={"/results": volume},
    gpu="L4",
    cpu=4,
    memory=16384,
    timeout=1200,
)
def profile_passes(run_id: str, expected_source: str):
    import benchmark as b
    import torch

    volume.reload()
    if source_hash() != expected_source:
        raise RuntimeError("uploaded source mismatch")
    root = Path("/results") / run_id
    report = json.loads((root / "results.json").read_text())
    if report["protocol"]["source_identity"] != b.source_identity():
        raise RuntimeError("library commit/version differs from training")
    for path, checksum in report["protocol"]["source_manifest"].items():
        if path.startswith("src/") and snapshot().get(path) != checksum:
            raise RuntimeError(f"library source differs from training: {path}")
    config = b.Config(**report["protocol"]["config"])
    packet = torch.load(
        Path("/results") / report["cloud"]["prepared"],
        map_location="cpu",
        weights_only=True,
    )
    b.configure("cuda")
    indices = b.epoch_order(config.train_size, config.seed, 0)[: config.batch_size]
    images, labels = (
        packet["train_images"][indices].cuda(),
        packet["train_labels"][indices].cuda(),
    )
    profiling = {
        "source_identity": b.source_identity(),
        "source_manifest": snapshot(),
        "batch_size": len(indices),
        "steps_per_pass": config.steps,
        "warmups_per_condition": 5,
        "repetitions_per_condition": 50,
        "environment": {
            "torch": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(),
        },
        "scope": "CUDA-synchronized wall latency; forward includes fresh state; backward excludes forward/loss preparation; no Adam or transfers; randomized interleaving on one L4",
        "states": {},
    }
    for state in ("initial", "snntorch_trained", "snnlab_trained"):
        if state == "initial":
            weights = packet["weights"]
            origin = "shared initial weights"
        else:
            backend = state.removesuffix("_trained")
            path = Path("/results") / report["cloud"]["checkpoints"][backend]
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if checkpoint["epoch"] != config.epochs:
                raise RuntimeError("incomplete training checkpoint")
            weights = checkpoint["parameters"]
            origin = {
                "backend": backend,
                "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        models = {
            "snntorch": b.TorchNetwork(config, weights).cuda(),
            "snnlab": b.LabNetwork(config, weights).cuda(),
        }
        for model in models.values():
            for key, parameter in model.canonical_parameters().items():
                torch.testing.assert_close(
                    parameter.detach().cpu(), weights[key], atol=0, rtol=0
                )
        profiling["states"][state] = {
            "weight_source": origin,
            "agreement": b.compare_outputs(models, images, labels),
            "timings": b.measure_passes(models, images, labels),
        }
        print(f"Completed {state}", flush=True)
    profiling["complete"] = True
    report["profiling"] = profiling
    b.write_json(root / "results.json", report)
    volume.commit()
    return json.dumps(report)


@app.local_entrypoint()
def main(
    mode: str = "collect",
    run_id: str = "mnist-30fabde19a1a7aa1",
    output: str = str(HERE / "results"),
):
    if mode == "train":
        run_id = "mnist-" + source_hash()[:16]
        result = json.loads(train_pair.remote(run_id, source_hash()))
    elif mode == "profile":
        result = json.loads(profile_passes.remote(run_id, source_hash()))
    elif mode == "collect":
        result = json.loads(b"".join(volume.read_file(f"/{run_id}/results.json")))
    else:
        raise ValueError("mode must be train, profile, or collect")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    (root / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    python = ROOT / ".venv/bin/python"
    if python.exists():
        subprocess.run(
            [str(python), str(HERE / "plot.py"), str(root)],
            check=True,
            env={**os.environ, "MPLCONFIGDIR": "/tmp/snnlab-matplotlib"},
        )
    print(f"Saved {root}; run_id={run_id}")
