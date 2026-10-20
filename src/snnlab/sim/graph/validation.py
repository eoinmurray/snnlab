"""Keep finite checks named while moving their host reads outside CUDA capture."""

from contextlib import contextmanager
from contextvars import ContextVar

import torch

_PENDING = ContextVar("snnlab_finite_checks", default=None)


@contextmanager
def deferred_validation():
    checks = []
    token = _PENDING.set(checks)
    try:
        yield checks
    finally:
        _PENDING.reset(token)


def check_condition(flag, message):
    pending = _PENDING.get()
    if pending is not None:
        pending.append((message, flag))
    elif not flag:
        raise ValueError(message)


def check_finite(value, name):
    check_condition(torch.isfinite(value).all(), f"{name} tensor must be finite")


def validate_pending(checks):
    if not checks:
        return
    by_device = {}
    for name, flag in checks:
        by_device.setdefault(flag.device, []).append((name, flag))
    for values in by_device.values():
        if not torch.stack([flag for _, flag in values]).all():
            for name, flag in values:
                if not flag:
                    raise ValueError(name)
