"""Scoped accelerator precision and automatic fused optimizer selection."""

from contextlib import contextmanager
from threading import RLock

import torch

_PRECISION_LOCK = RLock()


def amp_enabled(policy, parameter):
    if (
        policy != "auto"
        or parameter is None
        or parameter.device.type != "cuda"
        or parameter.dtype != torch.float32
    ):
        return False
    with torch.cuda.device(parameter.device):
        return torch.cuda.is_bf16_supported(including_emulation=False)


@contextmanager
def execution_precision(policy, parameter):
    if parameter is None or parameter.device.type != "cuda":
        yield
        return
    # PyTorch exposes matmul precision as process state. Serialize our scoped
    # changes and restore the caller's setting, including on execution failure.
    with _PRECISION_LOCK:
        previous = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("high" if policy == "auto" else "highest")
        try:
            with torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
                enabled=amp_enabled(policy, parameter),
                cache_enabled=False,
            ):
                yield
        finally:
            torch.set_float32_matmul_precision(previous)


def automatic_fused_optimizer(optimizer):
    """Select fused Adam implementations only when the caller left it unset."""
    if type(optimizer) not in {torch.optim.Adam, torch.optim.AdamW}:
        return False
    for group in optimizer.param_groups:
        if group.get("fused") is not None or group.get("foreach") is not None:
            continue
        if group.get("differentiable"):
            continue
        parameters = group["params"]
        if parameters and all(
            value.device.type == "cuda"
            and value.dtype
            in {torch.float16, torch.bfloat16, torch.float32, torch.float64}
            for value in parameters
        ):
            group["fused"] = True
    return any(group.get("fused", False) for group in optimizer.param_groups)
