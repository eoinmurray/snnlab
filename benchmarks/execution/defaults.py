"""Run unchanged benchmark workloads under before/automatic execution policies."""

import argparse
import importlib
import json
import sys
from dataclasses import replace
from functools import wraps
from pathlib import Path

from snnlab.sim import Performance

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        choices=("scikit-digits", "snntorch-comparison", "shd"),
        required=True,
    )
    parser.add_argument(
        "--policy", choices=("before", "after", "snntorch"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cpu", help="Device for scikit-digits")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "benchmarks" / args.benchmark))
    b = importlib.import_module("benchmark")
    if args.policy == "before":
        original = b.train

        @wraps(original)
        def legacy_train(**kwargs):
            policy = kwargs.pop("performance", None)
            legacy = {
                name: kwargs.pop(name)
                for name in Performance.__dataclass_fields__
                if name in kwargs
            }
            if policy is not None and legacy:
                raise TypeError("benchmark mixes performance with legacy keywords")
            policy = policy if policy is not None else Performance(**legacy)
            kwargs["performance"] = replace(
                policy,
                precision="highest",
                compile_block_steps=1,
                cuda_graphs="off",
                fused_optimizer="off",
                dataset_prefetch=policy.dataset_prefetch
                if args.benchmark == "shd"
                else 0,
                compile_mode=None
                if args.benchmark == "snntorch-comparison"
                else policy.compile_mode,
            )
            return original(**kwargs)

        b.train = legacy_train
    if args.benchmark == "scikit-digits":
        result = b.run(b.Config(), args.output, smoke=args.smoke, device=args.device)
    elif args.benchmark == "shd":
        from shd_recipe import Config

        cfg = replace(Config(), epochs=2, steps=64) if args.smoke else Config()
        result = b.run(
            cfg, args.output, Path("/shd/cache"), device="cuda:0", smoke=args.smoke
        )
    else:
        import torch

        config = (
            b.Config(train_size=256, test_size=256, hidden=32, steps=9, epochs=2)
            if args.smoke
            else b.Config()
        )
        prepared = args.output.parent / "prepared.pt"
        if not prepared.exists():
            b.prepare(config, "/mnist/mnist", prepared)
        packet = torch.load(prepared, map_location="cpu", weights_only=True)
        args.output.mkdir(parents=True, exist_ok=False)
        backend = "snntorch" if args.policy == "snntorch" else "snnlab"
        result = b.run_training(
            backend, config, packet, args.output / f"{backend}.pt", "cuda:0"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run.json").write_text(
        json.dumps(
            {
                "benchmark": args.benchmark,
                "policy": args.policy,
                "smoke": args.smoke,
                "result": result,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
