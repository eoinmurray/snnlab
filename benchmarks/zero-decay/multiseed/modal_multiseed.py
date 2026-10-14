"""Repeat the original, unmatched training comparison on paired seeds."""

import hashlib
import json
import os
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2] if modal.is_local() else Path("/work")
SOURCE = Path(os.environ.get("SNNLAB_COMPARISON_SOURCE", ROOT))
SOURCE_COMMIT = "69bb7013d1f4ed6826014283dcdf68e690166e54"
SEEDS = (17, 42, 123, 314, 2026)
BACKENDS = ("snntorch", "strict_full", "fast_full", "fast_outputs")
app = modal.App("snnlab-original-multiseed")
volume = modal.Volume.from_name("snnlab-original-multiseed-012", create_if_missing=True)
if modal.is_local():
    BENCHMARK_SHA = hashlib.sha256(
        (SOURCE / "benchmarks/zero_decay.py").read_bytes()
    ).hexdigest()
    expected = json.loads((HERE / "results.json").read_text())
    if BENCHMARK_SHA != expected["benchmark_sha256"]:
        raise ValueError(
            "Set SNNLAB_COMPARISON_SOURCE to the extracted original release source"
        )
    FROZEN_SOURCE_SHA = expected["frozen_source_sha256"]
    for relative, digest in FROZEN_SOURCE_SHA.items():
        if hashlib.sha256((SOURCE / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen release source differs: {relative}")
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "torch==2.10.0",
            "torchvision==0.25.0",
            "numpy",
            "snntorch==0.9.4",
            "scipy",
            "matplotlib",
            "scikit-learn",
            "h5py",
        )
        .add_local_dir(SOURCE / "src", "/work/src", ignore=["**/__pycache__/**"])
        .add_local_file(SOURCE / "benchmarks/zero_decay.py", "/work/zero_decay.py")
    )
else:
    BENCHMARK_SHA = hashlib.sha256(Path("/work/zero_decay.py").read_bytes()).hexdigest()
    image = None


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=16384,
    timeout=600,
    max_containers=4,
    volumes={"/results": volume},
)
def trial(seed, backend):
    import sys

    sys.path[:0] = ["/work", "/work/src"]
    import snntorch
    import torch
    import zero_decay as benchmark
    from torchvision.datasets import MNIST

    cache = Path("/results") / BENCHMARK_SHA / f"seed-{seed}" / f"{backend}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    weights = {
        name: value.cuda() for name, value in benchmark.initial_weights(seed).items()
    }
    if backend == "snntorch":
        model = benchmark.TorchModel(weights).cuda()
    else:
        model = benchmark.GraphModel(
            weights,
            "strict" if backend == "strict_full" else "fast",
            backend != "fast_outputs",
        ).cuda()
    train = MNIST("/tmp/mnist", train=True, download=True)
    test = MNIST("/tmp/mnist", train=False, download=True)
    print(f"Starting seed {seed}, {backend}", flush=True)
    metrics, final_weights = benchmark.pipeline(
        model,
        train.data.reshape(-1, 784).cuda().float() / 255,
        train.targets.cuda(),
        test.data.reshape(-1, 784).cuda().float() / 255,
        test.targets.cuda(),
        epochs=2,
        batch=128,
        seed=seed,
    )

    def weight_digest(values):
        digest = hashlib.sha256()
        for name, tensor in sorted(values.items()):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    row = {
        "seed": seed,
        "backend": backend,
        "metrics": metrics,
        "initial_weights_sha256": weight_digest(weights),
        "final_weights_sha256": weight_digest(final_weights),
        "torch": str(torch.__version__),
        "snntorch": snntorch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "benchmark_sha256": BENCHMARK_SHA,
        "source_commit": SOURCE_COMMIT,
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".tmp")
    temporary.write_text(json.dumps(row, indent=2) + "\n")
    temporary.replace(cache)
    volume.commit()
    print(
        f"Completed seed {seed}, {backend}: {metrics['test_accuracy']:.2%}", flush=True
    )
    return row


@app.local_entrypoint()
def main():
    output = HERE / "results.json"
    report = {
        "source_commit": SOURCE_COMMIT,
        "frozen_source_sha256": FROZEN_SOURCE_SHA,
        "benchmark_sha256": BENCHMARK_SHA,
        "seeds": SEEDS,
        "backends": BACKENDS,
        "protocol": {
            "architecture": [784, 256, 10],
            "steps": 50,
            "epochs": 2,
            "batch_size": 128,
            "train_samples": 60000,
            "test_samples": 10000,
            "alpha": 0.8,
            "beta": 0.9,
            "dtype": "float32",
            "tf32": False,
            "encoding": "Bernoulli(feature*0.25)",
            "optimizer": "Adam lr=0.001",
            "gradient_clip_norm": 1,
            "matched_backward_intervention": False,
            "seed_varies": "initial weights, minibatch order and train/test encoding; paired identically across backends",
            "scope": "accuracy repetition; concurrent trials are not a performance comparison",
        },
        "complete": False,
        "trials": [],
    }
    started = time.time()
    for row in trial.starmap(
        [(seed, backend) for seed in SEEDS for backend in BACKENDS],
        order_outputs=False,
    ):
        report["trials"].append(row)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"Saved {len(report['trials'])}/20 trials: seed {row['seed']} {row['backend']} {row['metrics']['test_accuracy']:.2%}",
            flush=True,
        )
    report["complete"] = True
    report["elapsed_s"] = time.time() - started
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Complete: {output}")
