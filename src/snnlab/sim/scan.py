"""Scan-mode utilities.

Provides shared utilities for scanning parameters and recording-key resolution.
"""

from __future__ import annotations

import logging

import torch

log = logging.getLogger("cli")


# ── Shared utilities (imported by train/infer/__main__) ──────────
def _auto_device() -> torch.device:
    """Auto-detect the fastest available compute device.

    Priority: CUDA (Nvidia GPU) > MPS (Apple Silicon) > CPU.

    Called when the user doesn't explicitly set --device on the CLI. This provides
    sensible defaults: use GPU if available (training is ~3x faster), fall back to
    CPU on machines without accelerators.

    The PINGLAB_DEVICE env var forces a specific device (e.g. "cpu", "mps",
    "cuda") when set, overriding auto-detection — useful for CPU/MPS parity
    checks and for pinning a device without a CLI flag.

    Returns:
        torch.device: cuda, mps, or cpu, in order of speed.
    """
    import os
    forced = os.environ.get("PINGLAB_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Recording-key resolvers (shared with cli.py, config.py, and notebooks) ──
# WHY THESE EXIST: Recording dicts use variable key names depending on network
# architecture (single-layer 'hid' vs multi-layer 'hid_1', 'hid_2', etc.).
# Snapshots and notebooks always need to find the DEEPEST/PRIMARY hidden and
# inhibitory spike trains. These functions abstract that lookup.

def primary_hid_key(rec):
    """Return the recording key for the deepest hidden layer.

    WHY: Multi-layer networks record each layer separately ('hid_0', 'hid_1', etc.).
    For analysis, we typically care about the DEEPEST/FINAL hidden layer
    (the one with the most relevant representation before readout).

    Returns:
        str: Recording key for the deepest excitatory layer.
             Single-layer models: 'hid'
             Multi-layer models: 'hid_1' (layer 1 is the deepest; layer 0 is input→hid_0)
             Networks with no hidden layer: 'hid' (default fallback)

    Args:
        rec: Recording dict from network.spike_record
    """
    hid_keys = sorted(k for k in rec if k.startswith("hid"))
    return hid_keys[-1] if hid_keys else "hid"


def primary_inh_key(rec):
    """Return the recording key for the deepest inhibitory layer, or None if no inhibition.

    WHY: Like primary_hid_key, but for inhibitory spikes. Networks without E-I
    structure return None (e.g., standard feedforward networks).

    Returns:
        str or None: Recording key for the deepest inhibitory layer (e.g. 'inh', 'inh_1'),
                     or None if the network doesn't have inhibitory populations.

    Args:
        rec: Recording dict from network.spike_record
    """
    inh_keys = sorted(k for k in rec if k.startswith("inh"))
    return inh_keys[-1] if inh_keys else None
