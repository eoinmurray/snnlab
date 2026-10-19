"""Lossless NumPy storage for tensor dtypes, including bfloat16 bit patterns."""

import numpy as np
import torch

BFLOAT16_STORAGE = "snnlab.bfloat16-bits/v1"


def numpy_payload(value):
    value = value.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        return value.view(torch.uint16).numpy(), {"tensor_storage": BFLOAT16_STORAGE}
    return value.numpy(), {}


def tensor_payload(array, row, *, label):
    if not isinstance(array, np.ndarray):
        raise ValueError(f"{label} must contain a NumPy array")
    storage = row.get("tensor_storage")
    if storage is not None and (
        storage != BFLOAT16_STORAGE or row["dtype"] != "bfloat16"
    ):
        raise ValueError(f"{label}: unsupported tensor storage policy {storage!r}")
    expected_dtype = "uint16" if storage == BFLOAT16_STORAGE else row["dtype"]
    if list(array.shape) != row["shape"] or str(array.dtype) != expected_dtype:
        raise ValueError(f"{label} metadata does not match tensors.npz")
    value = torch.from_numpy(array.copy())
    return value.view(torch.bfloat16) if storage == BFLOAT16_STORAGE else value
