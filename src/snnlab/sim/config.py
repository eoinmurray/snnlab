"""Configuration, network construction, and simulation runners.

Contains the Config dataclass, extract_weights, run_sim, and
backward-compat module globals.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from snnlab.sim import models as M
from snnlab.sim.inputs import (
    make_step_drive,
)
from snnlab.sim.models import COBANet
from snnlab.sim.timing import duration_steps

# =============================================================================
# Config
# =============================================================================


@dataclass
class Config:
    n_e: int = 1024
    n_i: int = 256
    seed: int = 42
    sim_ms: float = 600.0
    refractory_e_ms: float = 3.0
    refractory_i_ms: float = 1.5
    refractory_policy: str = "nearest"
    step_on_ms: float = 200.0
    step_off_ms: float = 300.0
    t_e_async: float = 0.0006
    sigma_e: float = 0.05
    w_ei: tuple = (0.5, 0.05)
    w_ie: tuple = (1.0, 0.1)
    w_ee: tuple = (0.0, 0.0)
    w_ii: tuple = (0.0, 0.0)
    recurrent_initial_zero_fraction: float = 0.2
    noise_sigma: float = 0.001
    noise_tau: float = 3.0
    spike_rate_base: float = 10.0
    w_in_spikes: tuple = (0.3, 0.06)
    w_in_i_spikes: tuple | None = None
    w_in_initial_zero_fraction: float = 0.95
    bias: float = 0.0002
    ei_ratio: float = 2.0
    device: str = "cpu"
    artifact_root: str = ""

    @property
    def torch_device(self):
        return torch.device(self.device)


cfg = Config()


def setup_model_globals(hidden_sizes):
    """Initialize M module globals from hidden_sizes.

    Sets M.N_HID, M.N_INH (computed as N_HID // 4), and M.HIDDEN_SIZES.
    Call this before building/loading networks to ensure module globals are consistent.

    WHY: The models.py module uses global state (M.N_HID, M.N_INH, M.HIDDEN_SIZES)
    during forward() and weight initialization. These must be set BEFORE the network
    is built or weights are loaded, otherwise the network's internal size assumptions
    will be wrong. This function centralizes the init logic (was duplicated in 4 places).

    Args:
        hidden_sizes: List of hidden layer sizes. E.g. [256] for 1 layer, [128, 256]
                      for 2 layers. Uses the LAST element as the primary N_HID.

    Side effects:
        Mutates: M.N_HID (= hidden_sizes[-1]), M.N_INH (= N_HID // 4),
                 M.HIDDEN_SIZES (= list copy of hidden_sizes)
    """
    hidden_sizes = list(hidden_sizes) if hidden_sizes else [256]
    M.N_HID = hidden_sizes[-1]
    M.N_INH = hidden_sizes[-1] // 4
    M.HIDDEN_SIZES = list(hidden_sizes)


def set_sim_dt(dt, t_ms):
    """Set the module-global timestep and derived step count that forward() reads.

    THE SINGLE CHOKE POINT for dt. Every entry point that calls COBANet.forward()
    MUST call this (with the run's dt and total sim duration) before the forward
    pass, or the network silently runs at the models.py module defaults
    (dt = 0.25 ms, T_steps derived from the 1000 ms default T_ms).

    WHY this exists (and why it is one function, not inlined):
    - forward() computes every dt-dependent constant LOCALLY from the module-global
      `dt` — decay_ampa/gaba, beta_snn/out, ref_steps (see models.forward).
      This is a deliberate torch.compile specialization choice: the constants live
      in the graph, specialized on dt, rather than as pre-computed globals.
    - It also reads the module-global `T_steps` for the integration-loop length and
      the recording-buffer allocation.
    - So the ONLY things an entry point must pin are M.dt, M.T_ms, M.T_steps.
    - Historically patch_dt/recompute_dt_constants did this centrally. Commit
      8befe44 removed that helper and inlined the replacement into infer.py's four
      functions ONLY — train.py and run_sim/run_sim_batch were missed, so any
      training or plain sim at dt != 0.25 silently ran at dt = 0.25 (wrong decays)
      and the wrong step count. Re-centralizing here kills that whole class of bug:
      new forward-calling code calls set_sim_dt and cannot forget a global.

    Args:
        dt:   Integration timestep in ms (e.g. 0.1 for the trained baselines).
        t_ms: Total simulation duration in ms for one trial.

    Side effects:
        Mutates: M.dt (= dt), M.T_ms (= t_ms), M.T_steps (whole trial steps).
    """
    steps = duration_steps(t_ms, dt)
    if steps < 1:
        raise ValueError("simulation duration must contain at least one timestep")
    M.dt = float(dt)
    M.T_ms = float(t_ms)
    M.T_steps = steps


def save_selected_npz(path, arrays, fields=None):
    """Write an explicit field selection losslessly; preserve existing defaults."""
    if fields is None:
        np.savez(path, **arrays)
        return
    metadata = {"dt", "n_e", "n_i", "T", "n_trials", "label", "recording_start_step"}
    keep = set(fields) | metadata
    selected = {k: v for k, v in arrays.items() if k in keep}
    if not set(selected) - metadata:
        raise ValueError("output field selection contains no available data arrays")
    np.savez_compressed(path, **selected)


def save_snapshot_npz(
    out_path,
    rec,
    dt,
    n_e,
    n_i,
    display=None,
    primary_hid_key_fn=None,
    primary_inh_key_fn=None,
    label=None,
    extra=None,
    output_fields=None,
):
    """Save spike recording and metadata to NPZ file for notebook analysis.

    SINGLE SOURCE OF TRUTH: All snapshot saving across train/infer/sim paths must
    use this function (was previously duplicated in ~4 places with inconsistent field
    names and missing metadata). Ensures all snapshots have consistent structure.

    WHY: Recording dicts from different paths use different key names:
    - train.py records to 'hid', 'inh', 'input' keys
    - Multi-layer networks record to 'hid_0', 'hid_1', etc.
    - Notebooks always expect 'spk_e', 'spk_i' (excitatory/inhibitory spikes)
    This function handles all the mapping and ensures notebooks always get consistent
    field names.

    Args:
        out_path: Path to output NPZ file
        rec: Recording dict from network.spike_record with spike tensors keyed by
             layer name (e.g. 'hid', 'inh', 'hid_0', 'hid_1', 'input')
        dt: Timestep (ms) — stored as metadata
        n_e: Number of excitatory neurons — stored as metadata
        n_i: Number of inhibitory neurons — stored as metadata
        output_fields: Optional retained-field names. Selected outputs are compressed
             losslessly. Population spike counts and active-cell traces can replace
             full snapshots; the default still emits every recorded field.
        display: Optional stimulus array (ext_g or input_spikes tensor).
                 Used as fallback for 'input_spikes' field if rec doesn't contain it.
        primary_hid_key_fn: Function that finds the deepest hidden layer key in rec.
                           Defaults to scan.primary_hid_key (handles multi-layer).
        primary_inh_key_fn: Function that finds the deepest inhibitory layer key in rec.
                           Defaults to scan.primary_inh_key.

    Output NPZ fields:
        Metadata:
        - dt (float32): Timestep (ms)
        - n_e (int32): Number of excitatory neurons
        - n_i (int32): Number of inhibitory neurons

        Spike data (canonicalized names):
        - spk_e: Excitatory spike raster (T, n_e) — from rec[hid_key]
        - spk_i: Inhibitory spike raster (T, n_i) — from rec[inh_key]
        - input_spikes: Input stimulus (T, n_in) — from rec['input'] or display arg

        All other recorded fields (voltages, conductances, etc.):
        - v_e_<layer>, v_i_<layer>, g_e_<layer>, g_i_<layer>, etc.
          (all other keys in rec, minus hid/inh/input)
    """
    from snnlab.sim.scan import primary_hid_key as _primary_hid_key
    from snnlab.sim.scan import primary_inh_key as _primary_inh_key

    if primary_hid_key_fn is None:
        primary_hid_key_fn = _primary_hid_key
    if primary_inh_key_fn is None:
        primary_inh_key_fn = _primary_inh_key

    # Start with metadata (dt, n_e, n_i are always present and consistent)
    npz_data = {
        "dt": np.float32(dt),
        "n_e": np.int32(n_e),
        "n_i": np.int32(n_i),
    }
    # Optional true class label of the snapshotted sample (for annotated rasters).
    if label is not None:
        npz_data["label"] = np.int32(label)

    # Resolve spike recording keys: find the deepest/primary hidden and inhibitory layers
    # primary_hid_key returns 'hid' for single-layer, 'hid_1' for the deepest multi-layer
    hid_key = primary_hid_key_fn(rec)
    inh_key = primary_inh_key_fn(rec)
    if hid_key not in rec:
        hid_key = None  # An explicitly I-only recording has no E trajectory.

    # Save excitatory spikes under canonical name 'spk_e' (from whatever key they're in)
    if hid_key:
        spk_e = rec[hid_key]
        # Convert torch tensors to numpy; handle numpy arrays directly
        npz_data["spk_e"] = spk_e.numpy() if hasattr(spk_e, "numpy") else spk_e

    # Save inhibitory spikes under canonical name 'spk_i' or create empty array if absent
    if inh_key:
        spk_i = rec[inh_key]
        npz_data["spk_i"] = spk_i.numpy() if hasattr(spk_i, "numpy") else spk_i
    else:
        # Networks without inhibition: save empty (T, 0) array so downstream code
        # doesn't break when trying to access spk_i
        T = npz_data.get("spk_e", rec[hid_key]).shape[0] if hid_key else 0
        npz_data["spk_i"] = np.zeros((T, 0), dtype=np.float32)

    # Save all other recorded fields (voltages, conductances, etc.) under their
    # original names, except 'input' which we rename to 'input_spikes' for clarity
    for key, val in rec.items():
        if key not in (hid_key, inh_key) and val is not None:
            # 'input' key is confusing; notebooks always expect 'input_spikes'
            save_key = "input_spikes" if key == "input" else key
            npz_data[save_key] = val.numpy() if hasattr(val, "numpy") else val

    # Fallback for input stimulus: if rec doesn't have 'input' or 'input_spikes',
    # and display was passed, save it as 'input_spikes'. This handles the
    # synthetic-spikes inference mode where we don't record the input to rec.
    if display is not None:
        display_arr = display.numpy() if hasattr(display, "numpy") else display
        # Only add if not already present (rec['input'] takes precedence)
        if "input_spikes" not in npz_data:
            npz_data["input_spikes"] = display_arr

    # Extra caller-supplied arrays (e.g. the Lyapunov ‖ΔV(t)‖ curve and its
    # time axis) written verbatim under their own keys.
    if extra is not None:
        for key, val in extra.items():
            if val is None:
                continue
            npz_data[key] = val.numpy() if hasattr(val, "numpy") else np.asarray(val)

    if output_fields is not None:
        for population in ("e", "i"):
            if not any(
                field in output_fields
                for field in (f"spk_{population}_count", f"v_{population}_selected")
            ):
                continue
            spikes = npz_data[f"spk_{population}"]
            if f"spk_{population}_count" in output_fields:
                npz_data[f"spk_{population}_count"] = np.asarray(
                    spikes.sum(), dtype=np.int64
                )
                npz_data["T"] = np.int64(spikes.shape[0])
            if f"v_{population}_selected" in output_fields:
                counts = spikes.sum(axis=0)
                index = int(np.argmax(counts)) if counts.size and counts.any() else 0
                npz_data[f"{population}_trace_index"] = np.int64(index)
                for signal in ("v", "ge", "gi"):
                    name = f"{signal}_{population}_1"
                    if name in npz_data:
                        npz_data[f"{signal}_{population}_selected"] = npz_data[name][
                            :, index
                        ]
        if "has_gi_e" in output_fields:
            npz_data["has_gi_e"] = np.asarray(npz_data["gi_e_1"].any())
    save_selected_npz(out_path, npz_data, output_fields)


# =============================================================================
# Model Registry
# =============================================================================

# Single source of truth for the model set: name → (class, base kwargs).
# build_net (train/infer) and _build_sim_net (sim/scan/snapshot) both read it.
_MODEL_CLASSES = {
    "ping": (COBANet, {}),
}


def _build_sim_net(model_name, spike_input=False, **kwargs):
    """Construct a network for the sim/scan/snapshot path from _MODEL_CLASSES.

    The ping path additionally wires the cfg-derived recurrent weight specs;
    the train/infer path sets those from CLI args via build_net instead.

    W_in: the conductance-drive path injects current directly onto E cells
    (ext_g), bypassing input synapses, so W_in is zeroed. When spike_input is
    True (synthetic-spikes: uniform Poisson fed THROUGH W_in), build real input
    synapses from cfg.w_in_spikes / cfg.w_in_initial_zero_fraction — otherwise the spikes hit
    a zero matrix and the network stays silent.
    """
    cls, base_kwargs = _MODEL_CLASSES[model_name]
    kwargs = {**base_kwargs, **kwargs}
    kwargs.setdefault("refractory_e_ms", cfg.refractory_e_ms)
    kwargs.setdefault("refractory_i_ms", cfg.refractory_i_ms)
    kwargs.setdefault("refractory_policy", cfg.refractory_policy)
    if model_name == "ping":
        w_in = (
            (*cfg.w_in_spikes, "lower_clamped_normal", cfg.w_in_initial_zero_fraction)
            if spike_input
            else (0, 0)
        )
        kwargs.update(
            w_in=w_in,
            w_in_i=(
                (
                    *cfg.w_in_i_spikes,
                    "lower_clamped_normal",
                    cfg.w_in_initial_zero_fraction,
                )
                if spike_input and cfg.w_in_i_spikes is not None
                else None
            ),
            w_hid=(5.1, 3.8),
            w_ee=(
                *cfg.w_ee,
                "lower_clamped_normal",
                cfg.recurrent_initial_zero_fraction,
            ),
            w_ei=(
                *cfg.w_ei,
                "lower_clamped_normal",
                cfg.recurrent_initial_zero_fraction,
            ),
            w_ie=(
                *cfg.w_ie,
                "lower_clamped_normal",
                cfg.recurrent_initial_zero_fraction,
            ),
        )
    return cls(**kwargs)


LEGACY_MODEL_ALIASES: dict[str, str] = {}


def build_net(
    model_name,
    w_in=None,
    w_in_i=None,
    w_in_initial_zero_fraction=0.0,
    w_ee=None,
    w_ei=None,
    w_ie=None,
    w_ii=None,
    ei_strength=None,
    ei_ratio=2.0,
    recurrent_initial_zero_fraction=0.0,
    device=None,
    randomize_init=False,
    dales_law=True,
    hidden_sizes=None,
    readout_mode="rate",
    signed_readout=False,
    readout_bias=False,
    readout_w_init=None,
    trainable_w_ee=False,
    trainable_w_ei=False,
    trainable_w_ie=False,
    trainable_w_ii=False,
    n_inh_per_layer=None,
    state_clamp=False,
    train_leak=False,
    tau_m_e_bounds_ms=None,
    tau_m_i_bounds_ms=None,
    adaptive_threshold=False,
    adapt_tau_bounds_ms=None,
    adapt_strength_init_mv=1.0,
    adapt_strength_max_mv=None,
    refractory_e_ms=None,
    refractory_i_ms=None,
    refractory_policy="nearest",
):
    """Construct a network with the given config.

    Single canonical builder used by every mode. Same args produce the same
    network — no drift between train/infer/image/sim.

    hidden_sizes: list of hidden layer sizes (e.g. [128, 256] for 2 layers).
    Every hidden layer gets E-I structure.
    """
    if model_name not in _MODEL_CLASSES:
        raise ValueError(
            f"Unknown model {model_name!r}; choose from {list(_MODEL_CLASSES)}"
        )
    cls, base_kwargs = _MODEL_CLASSES[model_name]
    kwargs = {**base_kwargs}
    kwargs["refractory_e_ms"] = refractory_e_ms
    kwargs["refractory_i_ms"] = refractory_i_ms
    kwargs["refractory_policy"] = refractory_policy
    kwargs["readout_mode"] = readout_mode
    kwargs["signed_readout"] = signed_readout
    kwargs["readout_bias"] = readout_bias
    kwargs["readout_w_init"] = readout_w_init
    kwargs["train_leak"] = train_leak
    kwargs["adaptive_threshold"] = adaptive_threshold
    kwargs["adapt_strength_init_mv"] = adapt_strength_init_mv
    if tau_m_e_bounds_ms is not None:
        kwargs["tau_m_e_bounds_ms"] = tuple(tau_m_e_bounds_ms)
    if tau_m_i_bounds_ms is not None:
        kwargs["tau_m_i_bounds_ms"] = tuple(tau_m_i_bounds_ms)
    if adapt_tau_bounds_ms is not None:
        kwargs["adapt_tau_bounds_ms"] = tuple(adapt_tau_bounds_ms)
    if adapt_strength_max_mv is not None:
        kwargs["adapt_strength_max_mv"] = float(adapt_strength_max_mv)

    # Set module-level hidden sizes
    if hidden_sizes is not None:
        M.HIDDEN_SIZES = list(hidden_sizes)
        M.N_HID = hidden_sizes[-1]
        kwargs["hidden_sizes"] = list(hidden_sizes)

    kwargs["dales_law"] = dales_law
    kwargs["state_clamp"] = state_clamp
    if trainable_w_ee:
        kwargs["trainable_w_ee"] = True
    if trainable_w_ei:
        kwargs["trainable_w_ei"] = True
    if trainable_w_ie:
        kwargs["trainable_w_ie"] = True
    if trainable_w_ii:
        kwargs["trainable_w_ii"] = True
    if w_in is not None:
        kwargs["w_in"] = (*w_in, "lower_clamped_normal", w_in_initial_zero_fraction)
    if w_in_i is not None:
        kwargs["w_in_i"] = (*w_in_i, "lower_clamped_normal", w_in_initial_zero_fraction)
    if w_ee is not None:
        kwargs["w_ee"] = (
            *w_ee,
            "lower_clamped_normal",
            recurrent_initial_zero_fraction,
        )
    if w_ei is not None:
        kwargs["w_ei"] = (
            *w_ei,
            "lower_clamped_normal",
            recurrent_initial_zero_fraction,
        )
    elif ei_strength is not None:
        s = ei_strength
        kwargs["w_ei"] = (
            s,
            s * 0.1,
            "lower_clamped_normal",
            recurrent_initial_zero_fraction,
        )
    if w_ie is not None:
        kwargs["w_ie"] = (
            *w_ie,
            "lower_clamped_normal",
            recurrent_initial_zero_fraction,
        )
    elif ei_strength is not None:
        s = ei_strength
        kwargs["w_ie"] = (
            s * ei_ratio,
            s * ei_ratio * 0.1,
            "lower_clamped_normal",
            recurrent_initial_zero_fraction,
        )
    if w_ii is not None:
        kwargs["w_ii"] = (
            *w_ii,
            "lower_clamped_normal",
            recurrent_initial_zero_fraction,
        )
    if (
        w_ei is None
        and w_ie is None
        and ei_strength is None
        and recurrent_initial_zero_fraction > 0
    ):
        kwargs.setdefault("initial_zero_fraction", recurrent_initial_zero_fraction)
    if n_inh_per_layer is not None:
        kwargs["n_inh_per_layer"] = dict(n_inh_per_layer)
    net = cls(**kwargs)
    if device is not None:
        net = net.to(device)
    if randomize_init and hasattr(net, "randomize_init"):
        setattr(net, "randomize_init", True)
    return net


# =============================================================================
# Backward-compat aliases (read from cfg, mutated in-place by CLI / scan fns)
# =============================================================================


# =============================================================================
# Device
# =============================================================================


# =============================================================================
# Simulation helpers
# =============================================================================


def _extract_records(net):
    """Convert a network's spike_record to a dict of numpy arrays (on CPU)."""
    rec = {}
    for k, v in net.spike_record.items():
        if isinstance(v, list):
            rec[k] = torch.stack(v).cpu().numpy()
        else:
            rec[k] = v.cpu().numpy() if hasattr(v, "cpu") else np.array(v)
    return rec


def extract_weights(net):
    """Extract weight arrays from a network for display.

    Handles both old-style named parameters (W_in, W_hid) and new-style
    ParameterList/Dict (W_ff, W_rec, W_ee, W_ei, W_ie).
    """
    weights = {}

    def _extract(w):
        if isinstance(w, nn.Parameter):
            return w.data.cpu().numpy().ravel()
        return w.cpu().numpy().ravel()

    # New-style: ParameterList W_ff
    if hasattr(net, "W_ff"):
        for i, w in enumerate(net.W_ff):
            if i == 0:
                weights["W_in"] = _extract(w)
            elif i == len(net.W_ff) - 1:
                weights["W_out"] = _extract(w)
            else:
                weights[f"W_ff_{i + 1}"] = _extract(w)

    # New-style: ParameterDict W_rec, W_ee, W_ei, W_ie
    for dict_name in ["W_rec", "W_ee", "W_ei", "W_ie"]:
        d = getattr(net, dict_name, None)
        if d is not None and isinstance(d, nn.ParameterDict) and len(d) > 0:
            if len(d) == 1:
                weights[dict_name] = _extract(list(d.values())[0])
            else:
                for k, w in d.items():
                    weights[f"{dict_name}_{k}"] = _extract(w)

    # Legacy fallback: old-style named params
    if not weights:
        for name in ["W_in", "W_hid", "W_rec", "W_ee", "W_ei", "W_ie"]:
            if hasattr(net, name):
                w = getattr(net, name)
                weights[name] = _extract(w)

    return weights


def run_sim(
    dt,
    t_e_ping,
    *,
    model_name="ping",
    t_e_async=None,
    input_spikes=None,
    input_spikes_i=None,
    ext_g=None,
    ext_g_i=None,
    ext_g_inhib_e=None,
    ext_g_inhib_i=None,
    v_perturb_eps=0.0,
    v_perturb_seed=0,
    recurrent_weight_scales=None,
    recording_mode="full",
):
    """Run a single simulation with any registered model.

    Drive sources (any combination may be supplied, they are additive inside
    the forward pass):
      - ``input_spikes``: (T, N_in) 0/1 raster routed through W_in onto the E
        cells (the synthetic-spikes input path).
      - ``input_spikes_i``: optional distinct (T, N_in) raster routed through
        W_in_i onto I; when omitted, input_spikes drives both populations.
      - ``ext_g`` / ``ext_g_i``: (T, N_E) / (T, N_I) per-cell excitatory
        conductance injected directly onto the E / I populations, bypassing
        W_in. This is the cell-drive path used by the V&S / Brunel balanced
        network experiments (nb050/nb058): per-cell independent Poisson,
        shared common Poisson, or frozen quenched-DC drive, all pre-built by
        the caller. Without it the network reverts to pure recurrent PING.
      - neither: the Börgers tonic-conductance step drive.

    ``v_perturb_eps`` > 0 kicks every membrane voltage by an ε-mV offset at
    t=0 (seeded by ``v_perturb_seed``) so a second pass on identical drive
    diverges only through the dynamics — the clone half of the Lyapunov probe.

    Returns (rec, ext_g_or_spikes_numpy, weights).
    """
    if t_e_async is None:
        t_e_async = cfg.t_e_async
    M.N_HID = cfg.n_e
    M.N_INH = cfg.n_i
    # Pin dt / T_ms / T_steps so forward() runs at the requested dt (not the 0.25
    # module default). Below, M.T_steps may be further clamped down to the actual
    # length of a supplied input/drive tensor.
    set_sim_dt(dt, cfg.sim_ms)
    T_steps = duration_steps(cfg.sim_ms, dt)

    def _to_dev(t):
        if t is None:
            return None
        t = t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float32)
        return t.to(cfg.torch_device)

    ext_g = _to_dev(ext_g)
    ext_g_i = _to_dev(ext_g_i)
    ext_g_inhib_e = _to_dev(ext_g_inhib_e)
    ext_g_inhib_i = _to_dev(ext_g_inhib_i)

    if input_spikes is not None:
        # Spike input (optionally with per-cell ext_g/ext_g_i on top).
        ext_g_tensor = None
        input_spikes = input_spikes.to(cfg.torch_device)
        if input_spikes_i is not None:
            input_spikes_i = input_spikes_i.to(cfg.torch_device)
        M.T_steps = min(M.T_steps, len(input_spikes))
        if input_spikes_i is not None:
            M.T_steps = min(M.T_steps, len(input_spikes_i))
        if ext_g is not None:
            M.T_steps = min(M.T_steps, len(ext_g))
        for conductance in (ext_g_i, ext_g_inhib_e, ext_g_inhib_i):
            if conductance is not None:
                M.T_steps = min(M.T_steps, len(conductance))
    elif ext_g is not None:
        # Pure cell-drive (no W_in input spikes) — e.g. quenched-DC probes.
        ext_g_tensor = None
        M.T_steps = min(M.T_steps, len(ext_g))
    else:
        ext_g_tensor, _ = make_step_drive(
            cfg.n_e,
            T_steps,
            dt,
            t_e_async,
            t_e_ping,
            cfg.step_on_ms,
            cfg.step_off_ms,
            cfg.sigma_e,
            cfg.noise_sigma,
            cfg.noise_tau,
            cfg.seed,
        )
        ext_g_tensor = ext_g_tensor.to(cfg.torch_device)
        M.T_steps = T_steps

    torch.manual_seed(cfg.seed)
    net = _build_sim_net(
        model_name,
        spike_input=input_spikes is not None,
        hidden_sizes=[M.N_HID],
    )
    net.to(cfg.torch_device)
    net.recording = True
    net.recording_mode = recording_mode

    fwd_kwargs: dict = {
        "v_perturb_eps": float(v_perturb_eps),
        "v_perturb_seed": int(v_perturb_seed),
        "recurrent_weight_scales": recurrent_weight_scales,
    }
    if input_spikes is not None:
        fwd_kwargs["input_spikes"] = input_spikes
        if input_spikes_i is not None:
            fwd_kwargs["input_spikes_i"] = input_spikes_i
        if ext_g is not None:
            fwd_kwargs["ext_g"] = ext_g
        if ext_g_i is not None:
            fwd_kwargs["ext_g_i"] = ext_g_i
        if ext_g_inhib_e is not None:
            fwd_kwargs["ext_g_inhib_e"] = ext_g_inhib_e
        if ext_g_inhib_i is not None:
            fwd_kwargs["ext_g_inhib_i"] = ext_g_inhib_i
    elif ext_g is not None:
        fwd_kwargs["ext_g"] = ext_g
        if ext_g_i is not None:
            fwd_kwargs["ext_g_i"] = ext_g_i
        if ext_g_inhib_e is not None:
            fwd_kwargs["ext_g_inhib_e"] = ext_g_inhib_e
        if ext_g_inhib_i is not None:
            fwd_kwargs["ext_g_inhib_i"] = ext_g_inhib_i
    else:
        fwd_kwargs["ext_g"] = ext_g_tensor

    with torch.no_grad():
        net.forward(**fwd_kwargs)

    rec = _extract_records(net)
    weights = extract_weights(net)

    if input_spikes is not None:
        display = input_spikes.cpu().numpy()
    elif ext_g is not None:
        display = ext_g.cpu().numpy()
    else:
        assert ext_g_tensor is not None
        display = ext_g_tensor.cpu().numpy()
    return rec, display, weights


# =============================================================================
# Config builder + globals sync
# =============================================================================


def build_config(args):
    """Build Config from CLI args."""
    c = Config()
    for name in ("refractory_e_ms", "refractory_i_ms"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(c, name, float(value))
    if getattr(args, "refractory_policy", None) is not None:
        c.refractory_policy = args.refractory_policy
    if getattr(args, "seed", None) is not None:
        c.seed = int(args.seed)
    if hasattr(args, "out_dir") and args.out_dir is not None:
        c.artifact_root = args.out_dir
    if hasattr(args, "n_hidden") and args.n_hidden is not None:
        # args.n_hidden may be an int or a list (multi-layer). For legacy Config,
        # use the last hidden size (the E-I / output-feeding layer).
        n_e = args.n_hidden[-1] if isinstance(args.n_hidden, list) else args.n_hidden
        c.n_e = n_e
        c.n_i = n_e // 4
    if hasattr(args, "drive") and args.drive is not None:
        c.t_e_async = args.drive
    c.ei_ratio = getattr(args, "ei_ratio", 2.0)
    ei_strength = getattr(args, "ei_strength", None)
    if ei_strength is not None:
        s = ei_strength
        c.w_ei = (s, s * 0.1)
        c.w_ie = (s * c.ei_ratio, s * c.ei_ratio * 0.1)
    if hasattr(args, "w_ei") and args.w_ei is not None:
        c.w_ei = tuple(args.w_ei)
    if hasattr(args, "w_ie") and args.w_ie is not None:
        c.w_ie = tuple(args.w_ie)
    if hasattr(args, "w_ee") and args.w_ee is not None:
        c.w_ee = tuple(args.w_ee)
    if hasattr(args, "w_in") and args.w_in is not None:
        w = args.w_in
        if len(w) == 1:
            w = [w[0], w[0] * 0.1]  # std = 10% of mean
        c.w_in_spikes = tuple(w[:2])
    if getattr(args, "w_in_i", None) is not None:
        c.w_in_i_spikes = tuple(args.w_in_i[:2])
    input_mode = getattr(args, "input", "synthetic-spikes")
    if hasattr(args, "n_in") and args.n_in is not None:
        M.N_IN = args.n_in
    elif hasattr(args, "n_input") and args.n_input is not None:
        # Compatibility for direct Config callers predating the CLI's n_in name.
        M.N_IN = args.n_input
    elif input_mode == "synthetic-spikes":
        M.N_IN = c.n_e
    recurrent_initial_zero_fraction = getattr(
        args, "recurrent_initial_zero_fraction", None
    )
    if recurrent_initial_zero_fraction is not None:
        c.recurrent_initial_zero_fraction = recurrent_initial_zero_fraction
    w_in_initial_zero_fraction = getattr(args, "w_in_initial_zero_fraction", None)
    if w_in_initial_zero_fraction is not None:
        c.w_in_initial_zero_fraction = w_in_initial_zero_fraction
    bias = getattr(args, "bias", None)
    if bias is not None:
        c.bias = bias
    # Honour --t-ms by syncing cfg.sim_ms; otherwise the scan path sizes the
    # input array from cfg.SIM_MS (default 600) while the network loop uses
    # args.t_ms, producing an off-by-one IndexError when they disagree.
    t_ms = getattr(args, "t_ms", None)
    if t_ms is not None:
        c.sim_ms = float(t_ms)
    # M.max_rate_hz is set by configure_models (the single models-globals boundary),
    # which runs for all modes after build_config — no need to set it here too.
    # Sync the module-level aliases here so callers can't forget to.
    _sync_globals_from_cfg(c)
    return c


def _sync_globals_from_cfg(c):
    """Install a Config as the module-wide source of truth.

    Every alias (C.N_E, C.W_EI, C.STEP_ON_MS, …) resolves to this cfg via
    the module __getattr__ below, so there is nothing else to mirror.
    """
    global cfg
    cfg = c


# Every config alias (C.N_E, C.W_EI, C.STEP_ON_MS, …) resolves to the live
# Config — one source of truth, no mirror globals, no sync step.
_CFG_ALIASES = {
    "DEVICE": "torch_device",
    "EI_RATIO": "ei_ratio",
    "NOISE_SIGMA": "noise_sigma",
    "NOISE_TAU": "noise_tau",
    "SEED": "seed",
    "SIGMA_E": "sigma_e",
    "SIM_MS": "sim_ms",
    "SPIKE_RATE_BASE": "spike_rate_base",
    "STEP_OFF_MS": "step_off_ms",
    "STEP_ON_MS": "step_on_ms",
    "T_E_ASYNC_DEFAULT": "t_e_async",
    "W_IN_INITIAL_ZERO_FRACTION": "w_in_initial_zero_fraction",
    "W_IN_SPIKES": "w_in_spikes",
    "N_E": "n_e",
    "N_I": "n_i",
    "W_EI": "w_ei",
    "W_IE": "w_ie",
    "RECURRENT_INITIAL_ZERO_FRACTION": "recurrent_initial_zero_fraction",
    "BIAS": "bias",
}


def __getattr__(name):
    field = _CFG_ALIASES.get(name)
    if field is not None:
        return getattr(cfg, field)
    if name == "ARTIFACT_ROOT":
        return Path(cfg.artifact_root)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
