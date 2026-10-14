import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2] if modal.is_local() else Path("/work")
app = modal.App("snnlab-accuracy-investigation")
if modal.is_local():
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
        .add_local_dir(ROOT / "src", "/work/src", ignore=["**/__pycache__/**"])
        .add_local_file(ROOT / "benchmarks" / "zero_decay.py", "/work/zero_decay.py")
    )
else:
    image = None


@app.function(image=image, gpu="L4", cpu=4, memory=16384, timeout=900)
def investigate():
    import os
    import sys

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    sys.path[:0] = ["/work", "/work/src"]
    import snntorch
    import torch
    import zero_decay as b
    from torchvision.datasets import MNIST

    import snnlab.sim.extensions as X

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    weights = {k: v.cuda() for k, v in b.initial_weights().items()}
    models = {
        "reference": b.TorchModel(weights).cuda(),
        "reference_twin": b.TorchModel(weights).cuda(),
        "original": b.GraphModel(weights, "strict", False).cuda(),
    }
    original = X.make_spike
    X.make_spike = lambda **kw: snntorch.surrogate.fast_sigmoid(slope=25)
    models["native_backward"] = b.GraphModel(weights, "strict", False).cuda()
    models["fast_native_backward"] = b.GraphModel(weights, "fast", False).cuda()
    X.make_spike = original
    import inspect
    import types

    forward_source = inspect.getsource(b.TorchModel.forward)
    forward_source = forward_source.replace(
        'self.values["input_bias"]',
        'torch.ones(frame.shape[0],1,device=frame.device) @ self.values["input_bias"]',
    )
    forward_source = forward_source.replace(
        'self.values["output_bias"]',
        'torch.ones(frame.shape[0],1,device=frame.device) @ self.values["output_bias"]',
    )
    import textwrap

    namespace = {"torch": torch}
    exec(textwrap.dedent(forward_source), namespace)
    models["matmul_bias_reference"] = b.TorchModel(weights).cuda()
    models["matmul_bias_reference"].forward = types.MethodType(
        namespace["forward"], models["matmul_bias_reference"]
    )
    optimizers = {
        k: torch.optim.Adam(m.parameters(), lr=0.001) for k, m in models.items()
    }
    dataset = MNIST("/tmp/mnist", train=True, download=True)
    features = dataset.data.reshape(-1, 784).cuda().float() / 255
    labels = dataset.targets.cuda()
    order = torch.randperm(60000, generator=torch.Generator().manual_seed(42)).cuda()
    report = {
        "snntorch": snntorch.__version__,
        "torch": str(torch.__version__),
        "steps": [],
        "first_score_difference": {},
        "first_weight_difference": {},
    }
    for step in range(40):
        start = step * 128
        ids = order[start : start + 128]
        events = b.encode(features[ids], 42 + start)
        records = {}
        for name, model in models.items():
            model.zero_grad(set_to_none=True)
            scores = model(events)
            loss = torch.nn.functional.cross_entropy(scores, labels[ids])
            loss.backward()
            grads = {
                k: v.grad.detach().clone()
                for k, v in model.parameters_by_name().items()
            }
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizers[name].step()
            records[name] = (
                scores.detach(),
                grads,
                b.copied_weights(model),
                float(loss.detach()),
                float(norm.detach()),
            )
        row = {"step": step}
        for name in (
            "reference_twin",
            "original",
            "native_backward",
            "fast_native_backward",
            "matmul_bias_reference",
        ):
            ref, cur = (
                records[
                    "matmul_bias_reference"
                    if name in ("native_backward", "fast_native_backward")
                    else "reference"
                ],
                records[name],
            )
            diff = int((ref[0] != cur[0]).sum())
            grad = max(float((ref[1][k] - cur[1][k]).abs().max()) for k in ref[1])
            weight = max(float((ref[2][k] - cur[2][k]).abs().max()) for k in ref[2])
            row[name] = {
                "different_scores": diff,
                "gradient_errors_by_name": {
                    k: float((ref[1][k] - cur[1][k]).abs().max()) for k in ref[1]
                },
                "max_gradient_error": grad,
                "max_weight_error": weight,
                "loss": cur[3],
                "gradient_norm": cur[4],
            }
            if diff and name not in report["first_score_difference"]:
                report["first_score_difference"][name] = step
            if weight and name not in report["first_weight_difference"]:
                report["first_weight_difference"][name] = step
        row["reference_loss"] = records["reference"][3]
        report["steps"].append(row)
        if step % 20 == 0:
            print(json.dumps(row), flush=True)
    return report


@app.function(image=image, gpu="L4", cpu=4, memory=16384, timeout=900)
def full_pipeline(backend):
    import os
    import sys

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    sys.path[:0] = ["/work", "/work/src"]
    import snntorch
    import torch
    import zero_decay as b
    from torchvision.datasets import MNIST

    import snnlab.sim.extensions as X

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    weights = {k: v.cuda() for k, v in b.initial_weights().items()}
    X.make_spike = lambda **kw: snntorch.surrogate.fast_sigmoid(slope=25)
    if backend == "fast_native_backward":
        model = b.GraphModel(weights, "fast", False).cuda()
    else:
        import inspect
        import textwrap
        import types

        source = inspect.getsource(b.TorchModel.forward)
        source = source.replace(
            'self.values["input_bias"]',
            'torch.ones(frame.shape[0],1,device=frame.device) @ self.values["input_bias"]',
        )
        source = source.replace(
            'self.values["output_bias"]',
            'torch.ones(frame.shape[0],1,device=frame.device) @ self.values["output_bias"]',
        )
        namespace = {"torch": torch}
        exec(textwrap.dedent(source), namespace)
        model = b.TorchModel(weights).cuda()
        model.forward = types.MethodType(namespace["forward"], model)
    data = MNIST("/tmp/mnist", train=True, download=True)
    test = MNIST("/tmp/mnist", train=False, download=True)
    row, end = b.pipeline(
        model,
        data.data.reshape(-1, 784).cuda().float() / 255,
        data.targets.cuda(),
        test.data.reshape(-1, 784).cuda().float() / 255,
        test.targets.cuda(),
        2,
        128,
    )
    return row


@app.local_entrypoint()
def main():
    result = investigate.remote()
    result["matched_full_pipelines"] = {}
    destination = HERE / "matched.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    for backend in ("matmul_bias_reference", "fast_native_backward"):
        result["matched_full_pipelines"][backend] = full_pipeline.remote(backend)
        destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "steps"}))
