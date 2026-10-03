"""Inference driver for the CLI.

Replays a trained network over a held-out dataset split, recomputes test
accuracy and rates, and optionally writes per-sample predictions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from snnlab.sim import models as M
from snnlab.sim.config import (
    build_net,
    save_selected_npz,
    save_snapshot_npz,
    set_sim_dt,
    setup_model_globals,
)
from snnlab.sim.datasets import DATASET_N_HIDDEN_DEFAULTS, load_dataset
from snnlab.sim.encoders import EVAL_SEED, encode_batch
from snnlab.sim.scan import _auto_device, primary_hid_key, primary_inh_key
from snnlab.sim.timing import duration_metadata, duration_steps, refractory_metadata
from snnlab.sim.train import seed_everything

log = logging.getLogger("cli")


def _hidden_sizes_from_state_dict(state):
    """Recover the hidden-layer E sizes from a saved COBANet state_dict.

    W_ff is the feed-forward ParameterList [input→H1, H1→H2, …, HN→out]; block i
    has shape (all_sizes[i], all_sizes[i+1]) where all_sizes = [N_IN, *hidden_sizes,
    N_OUT]. So each hidden size is the *output* dim of a W_ff block, for every block
    except the final readout (whose output is N_OUT). Recovering this lets infer()
    rebuild a network that matches the checkpoint instead of the dataset default —
    otherwise load_state_dict raises a shape mismatch on any non-default checkpoint.

    Returns None when the shape can't be inferred (no W_ff keys, e.g. a partial /
    transfer checkpoint, or a degenerate single-block net) so the caller falls back
    to its own default and stays in control.
    """
    idxs = sorted(
        int(k.split(".")[1])
        for k in state
        if k.startswith("W_ff.") and k.split(".")[1].isdigit()
    )
    if len(idxs) < 2:  # need at least one hidden block + the readout block
        return None
    sizes = [int(state[f"W_ff.{i}"].shape[1]) for i in idxs[:-1]]
    return sizes or None


def _pin_run(dt, t_ms, tau_gaba=None, seed=None):
    """Per-run global setup shared by every infer entry point.

    Seeds the RNGs (before any build/init randomness), pins dt / T_ms / T_steps
    through the single set_sim_dt choke point, and optionally overrides the GABA
    time constant. forward() derives decay_gaba from M.tau_gaba + M.dt each call,
    so setting the time constant is all that's needed.
    """
    seed_everything(seed)
    set_sim_dt(dt, t_ms)  # pin dt / T_ms / T_steps (single choke point)
    if tau_gaba is not None:
        M.tau_gaba = float(tau_gaba)


def _resolve_hidden_sizes(hidden_sizes, load_weights, default_sizes):
    """Pick the hidden-layer sizes, then install them into the M globals.

    Precedence: an explicit `hidden_sizes` wins; else recover them from the
    checkpoint (a trained weights.pth fully determines them, and rebuilding at any
    other size makes load_state_dict fail on shape mismatch); else fall back to
    `default_sizes`. Centralised so every entry point agrees — this derivation was
    previously copy-pasted into infer/snapshot/probe/dump and drifted. Returns the
    chosen list (also written to M via setup_model_globals).
    """
    if hidden_sizes is None and load_weights is not None:
        hidden_sizes = _hidden_sizes_from_state_dict(
            torch.load(load_weights, map_location="cpu")
        )
        if hidden_sizes is not None:
            log.info(f"  n_hidden ← checkpoint {hidden_sizes}")
    if hidden_sizes is None:
        hidden_sizes = list(default_sizes)
        log.info(f"  n_hidden auto → {hidden_sizes}")
    setup_model_globals(hidden_sizes)
    return hidden_sizes


def _make_perturb_fn(mode, level, dt_ms, generator, counts=None):
    """Per-step hidden-spike perturbation callback (s_e, s_i, layer) -> (s_e', s_i').

    A closed family of dynamics-faithful perturbations installed on
    net._hidden_perturb_fn (models.py runs it right after spikes are emitted, so
    the I-loop and readout react within the trial):
      - drop: Bernoulli mask, each spike kept with prob (1 - level)
      - add: Bernoulli insertion at nominal `level` Hz per cell, capped at one spike
    """
    import math

    import torch
    probability = level if mode == "drop" else level * dt_ms / 1000.0
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("perturbation probability must be between zero and one")

    def record(before, after, population, layer):
        if counts is not None:
            key = f"{population}{layer}"
            values = torch.stack((
                before.sum(dtype=torch.int64), after.sum(dtype=torch.int64),
                ((before == 0) & (after > 0)).sum(dtype=torch.int64),
                ((before > 0) & (after == 0)).sum(dtype=torch.int64),
            ))
            if key not in counts:
                counts[key] = {"counts": values, "slots": before.numel()}
            else:
                counts[key]["counts"].add_(values)
                counts[key]["slots"] += before.numel()

    if mode == "drop":
        def fn(s_e, s_i, _layer):
            before_e, before_i = s_e, s_i
            s_e = s_e * (torch.rand(s_e.shape, generator=generator, device=s_e.device) >= level).float()
            if s_i is not None:
                s_i = s_i * (torch.rand(s_i.shape, generator=generator, device=s_i.device) >= level).float()
            record(before_e, s_e, "e", _layer)
            if s_i is not None:
                record(before_i, s_i, "i", _layer)
            return s_e, s_i
        return fn

    if mode == "add":
        p = level * dt_ms / 1000.0
        def fn(s_e, s_i, _layer):
            before_e, before_i = s_e, s_i
            s_e = torch.clamp(s_e + (torch.rand(s_e.shape, generator=generator, device=s_e.device) < p).float(), 0.0, 1.0)
            if s_i is not None:
                s_i = torch.clamp(s_i + (torch.rand(s_i.shape, generator=generator, device=s_i.device) < p).float(), 0.0, 1.0)
            record(before_e, s_e, "e", _layer)
            if s_i is not None:
                record(before_i, s_i, "i", _layer)
            return s_e, s_i
        return fn

    raise ValueError(f"unknown perturbation mode {mode!r}")


def infer(
    model_name="ping",
    dt=0.25,
    load_weights=None,
    dataset="mnist",
    max_samples=None,
    t_ms=200.0,
    w_in=None,
    ei_strength=0.5,
    ei_ratio=2.0,
    w_in_initial_zero_fraction=0.0,
    recurrent_initial_zero_fraction=0.0,
    hidden_sizes=None,
    out_dir=None,
    dales_law=True,
    seed=None,
    outputs=None,
    tau_gaba=None,
    scale_w_in=1.0,
    scale_w_ei=1.0,
    scale_w_ie=1.0,
    skip_load=None,
    perturb_mode=None,
    perturb_level=None,
    i_override_file=None,
    readout_mode="rate",
    signed_readout=False,
    readout_bias=False,
    train_leak=False,
    tau_m_e_bounds_ms=None,
    tau_m_i_bounds_ms=None,
    adaptive_threshold=False,
    adapt_tau_bounds_ms=None,
    adapt_strength_init_mv=1.0,
    adapt_strength_max_mv=None,
    recording_mode="full",
    output_fields=None,
    refractory_e_ms=None,
    refractory_i_ms=None,
    refractory_policy="nearest",
):
    """Run inference with saved weights at a given dt.

    outputs: iterable of extra artifacts to emit from this one forward pass, from
    {"per_cell_rates", "pop_traces", "rasters"}. metrics.json is always written.
    Each maps to an emitter below; recording is enabled only if a recording-backed
    output is requested, so the default accuracy path stays cheap.
    recording_mode selects full traces (default), E/I spikes, or I spikes only.
    output_fields optionally selects a union of fields across requested NPZs;
    shape metadata is retained automatically and selected files are compressed.
    """
    outputs = set(outputs or ())
    emit_per_cell_rates = "per_cell_rates" in outputs
    emit_pop_traces = "pop_traces" in outputs
    emit_rasters = "rasters" in outputs
    if recording_mode not in ("full", "spikes", "inhibitory"):
        raise ValueError(f"unknown recording mode: {recording_mode}")
    if recording_mode == "inhibitory" and (emit_per_cell_rates or emit_pop_traces):
        raise ValueError("inhibitory recording supports rasters only")

    # Seed, pin dt, and resolve hidden_sizes (explicit > checkpoint > dataset
    # default) — the shared per-run setup (see _pin_run / _resolve_hidden_sizes).
    _pin_run(dt, t_ms, tau_gaba=tau_gaba, seed=seed)
    hidden_sizes = _resolve_hidden_sizes(
        hidden_sizes, load_weights, [DATASET_N_HIDDEN_DEFAULTS.get(dataset, 256)]
    )
    device = _auto_device()

    # MNIST inference reports the untouched official test partition. Training
    # uses a validation split drawn only from the official training partition.
    _, X_te, _, y_te = load_dataset(
        dataset,
        max_samples=None,
        split=True,
        evaluation_split="test",
        evaluation_only=True,
    )
    if max_samples is not None and max_samples < len(y_te):
        eval_idx = np.random.RandomState(42).choice(
            len(y_te), max_samples, replace=False
        )
        X_te, y_te = X_te[eval_idx], y_te[eval_idx]
    M.N_IN = 784

    from torch.utils.data import DataLoader, TensorDataset

    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)), batch_size=64
    )

    # Build model — same builder as train, same args produce same network
    # Uniform randomize_init across all models — symmetry breaking matters
    # for all architectures. Only skip when kaiming (already heterogeneous).
    net = build_net(
        model_name,
        refractory_e_ms=refractory_e_ms,
        refractory_i_ms=refractory_i_ms,
        refractory_policy=refractory_policy,
        w_in=w_in,
        w_in_initial_zero_fraction=w_in_initial_zero_fraction,
        recurrent_initial_zero_fraction=recurrent_initial_zero_fraction,
        ei_strength=ei_strength,
        ei_ratio=ei_ratio,
        device=device,
        randomize_init=True,
        dales_law=dales_law,
        hidden_sizes=hidden_sizes,
        readout_mode=readout_mode,
        signed_readout=signed_readout,
        readout_bias=readout_bias,
        train_leak=train_leak,
        tau_m_e_bounds_ms=tau_m_e_bounds_ms,
        tau_m_i_bounds_ms=tau_m_i_bounds_ms,
        adaptive_threshold=adaptive_threshold,
        adapt_tau_bounds_ms=adapt_tau_bounds_ms,
        adapt_strength_init_mv=adapt_strength_init_mv,
        adapt_strength_max_mv=adapt_strength_max_mv,
    )

    # Load weights. skip_load drops matching state_dict keys (by prefix) so a
    # freshly-initialised sub-block survives — e.g. transfer-loading W_ff/W_ee from
    # a COBA checkpoint while keeping a fresh ei_strength I-loop (skip W_ei/W_ie).
    assert load_weights is not None
    state = torch.load(load_weights, map_location=device)
    if skip_load:
        state = {
            k: v
            for k, v in state.items()
            if not any(k.startswith(p) for p in skip_load)
        }
        log.info(f"  skip-load: dropped keys matching {list(skip_load)}")
    net.load_state_dict(state, strict=False)
    log.info(f"  loaded {load_weights}")

    # Inference-time weight scaling. Multiply the loaded matrices in place before
    # the forward pass — used by coupling / W_in sweeps that probe how accuracy,
    # rate and loss respond to scaled recurrent or input weights without retraining.
    if scale_w_in != 1.0:
        net.W_ff[0].data.mul_(float(scale_w_in))
    if scale_w_ei != 1.0:
        for _k in net.W_ei:
            net.W_ei[_k].data.mul_(float(scale_w_ei))
    if scale_w_ie != 1.0:
        for _k in net.W_ie:
            net.W_ie[_k].data.mul_(float(scale_w_ie))
    if (scale_w_in, scale_w_ei, scale_w_ie) != (1.0, 1.0, 1.0):
        log.info(
            f"  scaled weights: W_in×{scale_w_in} W_ei×{scale_w_ei} W_ie×{scale_w_ie}"
        )

    # Optional hidden-spike perturbation: install the callback so drop/add noise
    # is applied inside the forward loop (I-loop + readout react within the trial).
    perturbation_counts = {}
    if perturb_mode is not None:
        _pgen = torch.Generator(device=device).manual_seed(EVAL_SEED + 1)
        net._hidden_perturb_fn = _make_perturb_fn(
            perturb_mode, perturb_level, dt, _pgen, perturbation_counts
        )
        log.info(f"  perturb: {perturb_mode} level={perturb_level}")

    # Evaluate — pre-encode pixels as Poisson spikes (same path as train)
    import torch.nn.functional as F

    net.eval()
    correct = total = 0
    eval_gen = torch.Generator().manual_seed(EVAL_SEED)
    # Accumulate per-population firing rate (Hz) across batches, weighted by
    # batch size. net.rates is set by _set_meta after each forward as Hz
    # per population averaged over the batch.
    rate_sums: dict[str, float] = {}

    # E1: per-cell rate emission. When requested, turn on spike recording and
    # accumulate per-cell spike counts (summed over time and batch) for the
    # primary E and I layers. Notebooks that need the per-cell rate distribution
    # (rather than the population mean) consume per_cell_rates.npz instead of
    # rebuilding the net in-process. Recording is off by default so the common
    # accuracy-only path stays cheap.
    # E2: population-trace emission. When requested, record the per-timestep mean
    # spike activity over cells for each test trial (the population signal every
    # PSD / f_gamma notebook reduces to). The CLI emits this base time-series and
    # the notebook computes the spectrum itself — no metric logic in the CLI.
    # rasters: sparse per-trial spike-index emission (base data for cycle-level
    # analyses — participation p, spikes-per-cell-per-cycle — that stream over the
    # full rasters). Spikes are sparse, so we store (trial, t, cell) indices, not
    # dense arrays. Any recording-backed output turns recording on.
    _recording_outputs = emit_per_cell_rates or emit_pop_traces or emit_rasters
    if _recording_outputs:
        net.recording = True
        net.recording_mode = recording_mode
    per_cell_e = None  # running (N_E,) spike-count sum across the test set
    per_cell_i = None  # running (N_I,) spike-count sum across the test set
    per_sample_e_rates: list = []  # one hidden-E population-mean rate per trial
    pop_e_rows: list = []  # per-trial (T,) mean-over-E-cells activity
    pop_i_rows: list = []  # per-trial (T,) mean-over-I-cells activity
    # rasters: flat COO lists (trial index, timestep, cell index) for E, I,
    # and the output LIF population when the selected readout emits spikes.
    rast_e = {"trial": [], "t": [], "cell": []}
    rast_i = {"trial": [], "t": [], "cell": []}
    rast_out = {"trial": [], "t": [], "cell": []}
    rast_trial = 0  # running global trial counter across batches
    rast_T = 0  # timesteps (set from the recorded raster)

    def _accum_rasters(rec, key, store, base_trial):
        """Append sparse (trial, t, cell) spike indices for one population."""
        r = rec.get(key) if key else None
        if r is None:
            return 0, 0
        arr = r.detach().cpu().numpy()
        if arr.ndim == 2:  # (T, N) → (T, 1, N)
            arr = arr[:, None, :]
        T, B, _N = arr.shape
        t_idx, b_idx, c_idx = arr.nonzero()
        store["trial"].append((base_trial + b_idx).astype("int32"))
        store["t"].append(t_idx.astype("int32"))
        store["cell"].append(c_idx.astype("int32"))
        return B, T

    def _accum_per_cell(rec, key, acc):
        """Sum a recorded raster over time (and batch) to per-cell counts."""
        r = rec.get(key) if key else None
        if r is None:
            return acc
        # (T, B, N) → sum over T and B; (T, N) at B=1 → sum over T.
        cnt = r.sum(dim=(0, 1)) if r.ndim == 3 else r.sum(dim=0)
        cnt = cnt.detach().cpu().numpy()
        return cnt if acc is None else acc + cnt

    def _accum_per_sample_rates(rec, key, rows):
        """Append population-mean firing rates (Hz), one value per trial."""
        r = rec.get(key) if key else None
        if r is None:
            return
        if r.ndim == 3:  # (T, B, N) -> (B,)
            counts = r.sum(dim=0).mean(dim=1)
        else:  # (T, N) -> scalar batch of one
            counts = r.sum(dim=0).mean().unsqueeze(0)
        rows.extend((counts / ((M.T_steps * dt) / 1000.0)).detach().cpu().tolist())

    def _pop_rows(rec, key, rows):
        """Append per-trial population activity: mean over cells at each timestep."""
        r = rec.get(key) if key else None
        if r is None:
            return
        # (T, B, N) → mean over N → (T, B); (T, N) at B=1 → mean over N → (T,).
        if r.ndim == 3:
            pop = r.mean(dim=2).detach().cpu().numpy()  # (T, B)
            for b in range(pop.shape[1]):
                rows.append(pop[:, b].astype("float32"))
        else:
            rows.append(r.mean(dim=1).detach().cpu().numpy().astype("float32"))

    # I-spike override: substitute the inhibitory spikes each timestep with a
    # pre-built per-trial stream (loaded from a sparse NPZ — the generic dual of
    # --outputs rasters). The notebook builds the override (jitter/shuffle/etc.)
    # from a baseline pass; the CLI just injects it. Per batch we reconstruct the
    # dense (B, T, N_I) slice for that batch's trials and install a stateful hook.
    _iov: dict | None = None  # heterogeneous: ints + arrays, keyed by name
    if i_override_file is not None:
        import numpy as _np

        z = _np.load(i_override_file)
        _iov = {
            "T": int(z["T"]),
            "n_i": int(z["n_i"]),
            "n_trials": int(z["n_trials"]),
        }
        _tr = z["i_trial"]
        _ord = _np.argsort(_tr, kind="stable")
        _iov["tr"] = _tr[_ord]
        _iov["t"] = z["i_t"][_ord]
        _iov["c"] = z["i_cell"][_ord]
        _iov["bounds"] = _np.searchsorted(_iov["tr"], _np.arange(_iov["n_trials"] + 1))
        _iov["g"] = 0  # running global trial offset
        # The override hook and online rates do not require trajectory recording.
        log.info(f"  i-override: {i_override_file} ({_iov['n_trials']} trials)")

    ce_sum = (
        0.0  # cross-entropy summed over samples → mean loss (a standard eval metric)
    )
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            spk = encode_batch(X_b, dt, generator=eval_gen)
            if _iov is not None:
                import numpy as _np

                Bc = y_b.size(0)
                ov = _np.zeros((Bc, _iov["T"], _iov["n_i"]), dtype="float32")
                for j in range(Bc):
                    tr = _iov["g"] + j
                    lo, hi = _iov["bounds"][tr], _iov["bounds"][tr + 1]
                    ov[j, _iov["t"][lo:hi], _iov["c"][lo:hi]] = 1.0
                _ov_t = torch.from_numpy(ov).to(device)  # (B, T, N_I)
                _st = {"step": 0}

                def _ov_fn(s_e, s_i, _layer, _ov=_ov_t, _st=_st):
                    t = _st["step"]
                    _st["step"] = t + 1
                    if s_i is None:
                        return s_e, s_i
                    return s_e, _ov[:, t, :]

                net._hidden_perturb_fn = _ov_fn
                _iov["g"] += Bc
            logits = net(input_spikes=spk)
            ce_sum += float(F.cross_entropy(logits, y_b, reduction="sum").item())
            correct += (logits.argmax(1) == y_b).sum().item()
            total += y_b.size(0)
            batch_rates = getattr(net, "rates", None) or {}
            B = y_b.size(0)
            for k, v in batch_rates.items():
                rate_sums[k] = rate_sums.get(k, 0.0) + float(v) * B
            if _recording_outputs:
                rec = net.spike_record
                hk, ik = primary_hid_key(rec), primary_inh_key(rec)
                if emit_per_cell_rates:
                    if output_fields is None or "rate_e_per_cell" in output_fields:
                        per_cell_e = _accum_per_cell(rec, hk, per_cell_e)
                    if output_fields is None or "rate_i_per_cell" in output_fields:
                        per_cell_i = _accum_per_cell(rec, ik, per_cell_i)
                    if output_fields is None or "rate_e_per_sample" in output_fields:
                        _accum_per_sample_rates(rec, hk, per_sample_e_rates)
                if emit_pop_traces:
                    if output_fields is None or "pop_e" in output_fields:
                        _pop_rows(rec, hk, pop_e_rows)
                    if output_fields is None or "pop_i" in output_fields:
                        _pop_rows(rec, ik, pop_i_rows)
                if emit_rasters:
                    nb_e, rast_T = _accum_rasters(rec, hk, rast_e, rast_trial)
                    nb_i, i_T = _accum_rasters(rec, ik, rast_i, rast_trial)
                    if output_fields is None or "out_t" in output_fields:
                        _accum_rasters(rec, "out_spikes", rast_out, rast_trial)
                    rast_T = rast_T or i_T
                    rast_trial += nb_e or nb_i

    acc = 100.0 * correct / total
    ce_loss = ce_sum / total if total else 0.0
    rates_hz = {k: v / total for k, v in rate_sums.items()} if total else {}
    hid_key = max((k for k in rates_hz if k.startswith("hid")), default=None)
    hid_rate_hz = rates_hz.get(hid_key) if hid_key else None
    rate_str = f"  hid={hid_rate_hz:.1f}Hz" if hid_rate_hz is not None else ""
    log.info(f"  dt={dt:.3f}ms  acc={acc:.1f}%  ({correct}/{total}){rate_str}")

    # Structured metrics artifact for tests/analysis (parallels train's metrics.json)
    out_dir_path = Path(out_dir) if out_dir else None
    if out_dir_path and out_dir_path.exists():
        from datetime import datetime, timezone

        metrics_blob = {
            "mode": "infer",
            "model": model_name,
            "run_finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": {
                "dt": dt,
                "t_ms": t_ms,
                **duration_metadata(t_ms, dt),
                **refractory_metadata(
                    net.refractory_e_ms, net.refractory_i_ms, dt,
                    policy=net.refractory_policy,
                ),
                "w_in": list(w_in) if w_in else None,
                "w_in_initial_zero_fraction": w_in_initial_zero_fraction,
                "ei_strength": ei_strength,
                "ei_ratio": ei_ratio,
                "n_hidden": M.N_HID,
                "n_inh": M.N_INH,
                "n_in": M.N_IN,
                "max_samples": max_samples,
                "dataset": dataset,
                "evaluation_partition": (
                    "official_mnist_test" if dataset == "mnist" else "official_shd_test"
                ),
                "evaluation_samples": total,
                "load_weights": str(load_weights),
            },
            "best_acc": acc,
            "ce_loss": ce_loss,
            "n_correct": correct,
            "n_total": total,
        }
        metrics_blob["rates_hz"] = rates_hz
        metrics_blob["hid_rate_hz"] = hid_rate_hz
        if perturb_mode is not None and i_override_file is None:
            metrics_blob["perturbation"] = {
                "mode": perturb_mode, "level": perturb_level, "dt_ms": dt,
                "populations": {
                    key: {
                        **dict(zip(("raw_spikes", "transmitted_spikes", "inserted_spikes", "deleted_spikes"),
                                   row["counts"].cpu().tolist())),
                        "slots": row["slots"],
                    } for key, row in perturbation_counts.items()
                },
            }
        with open(out_dir_path / "metrics.json", "w") as f:
            json.dump(metrics_blob, f, indent=2, default=float)
        log.info(f"  → {out_dir_path / 'metrics.json'}")

    # E1: write per-cell rate arrays if requested. rate = total spikes per cell
    # over the test set / (n_trials × physical trial time in seconds) → Hz.
    if emit_per_cell_rates and out_dir_path and out_dir_path.exists():
        t_sec = (M.T_steps * dt) / 1000.0
        denom = total * t_sec if total else 1.0
        dump = {}
        if per_cell_e is not None:
            dump["rate_e_per_cell"] = (per_cell_e / denom).astype(np.float32)
        if per_cell_i is not None:
            dump["rate_i_per_cell"] = (per_cell_i / denom).astype(np.float32)
        if per_sample_e_rates:
            dump["rate_e_per_sample"] = np.asarray(per_sample_e_rates, dtype=np.float32)
        out_npz = out_dir_path / "per_cell_rates.npz"
        save_selected_npz(out_npz, dump, output_fields)
        log.info(f"  → {out_npz}  ({len(dump)} arrays)")

    # E2: write per-trial population activity traces (base signal for PSD/f_gamma).
    # Each row is one test trial's per-timestep mean spike activity over cells.
    if emit_pop_traces and out_dir_path and out_dir_path.exists():
        dump = {"dt": np.float32(dt)}
        if pop_e_rows:
            dump["pop_e"] = np.stack(pop_e_rows)  # (n_samples, T)
        if pop_i_rows:
            dump["pop_i"] = np.stack(pop_i_rows)  # (n_samples, T)
        out_npz = out_dir_path / "pop_traces.npz"
        save_selected_npz(out_npz, dump, output_fields)
        n_e = dump.get("pop_e", np.zeros((0, 0))).shape
        log.info(f"  → {out_npz}  (pop_e={n_e})")

    # rasters: write sparse per-trial spike indices for cycle-level analyses.
    # Format: for each population, concatenated int32 (trial, t, cell) index
    # vectors, plus shape metadata so the notebook can reconstruct dense rasters
    # per trial. Spikes are sparse so this stays small (tens of MB) vs the dense
    # (n_trials × T × N) arrays it represents.
    if emit_rasters and out_dir_path and out_dir_path.exists():

        def _cat(store):
            if not store["trial"]:
                return (np.zeros(0, np.int32),) * 3
            return (
                np.concatenate(store["trial"]),
                np.concatenate(store["t"]),
                np.concatenate(store["cell"]),
            )

        e_tr, e_t, e_c = _cat(rast_e)
        i_tr, i_t, i_c = _cat(rast_i)
        out_tr, out_t, out_c = _cat(rast_out)
        out_npz = out_dir_path / "rasters.npz"
        populations = {"i_trial": i_tr, "i_t": i_t, "i_cell": i_c}
        if recording_mode != "inhibitory":
            populations.update(
                e_trial=e_tr,
                e_t=e_t,
                e_cell=e_c,
                out_trial=out_tr,
                out_t=out_t,
                out_cell=out_c,
            )
        save_selected_npz(
            out_npz,
            dict(
                dt=np.float32(dt),
                n_trials=np.int32(rast_trial),
                T=np.int32(rast_T),
                n_e=np.int32(M.N_HID),
                n_i=np.int32(M.N_INH),
                **populations,
            ),
            output_fields,
        )
        mb = out_npz.stat().st_size / 1e6
        log.info(
            f"  → {out_npz}  ({rast_trial} trials, "
            f"{e_tr.size + i_tr.size + out_tr.size} spikes, {mb:.1f} MB)"
        )

    return {
        "acc": acc,
        "ce_loss": ce_loss,
        "rates_hz": rates_hz,
        "hid_rate_hz": hid_rate_hz,
    }


def infer_and_snapshot(
    model_name="ping",
    dt=0.25,
    load_weights=None,
    dataset="mnist",
    t_ms=200.0,
    w_in=None,
    ei_strength=0.5,
    ei_ratio=2.0,
    w_in_initial_zero_fraction=0.0,
    recurrent_initial_zero_fraction=0.0,
    hidden_sizes=None,
    out_dir=None,
    dales_law=True,
    seed=None,
    digit=0,
    sample=0,
    sample_index=None,
    tau_gaba=None,
    skip_load=None,
    perturb_mode=None,
    perturb_level=None,
    i_override_file=None,
    readout_mode="rate",
    signed_readout=False,
    readout_bias=False,
    train_leak=False,
    tau_m_e_bounds_ms=None,
    tau_m_i_bounds_ms=None,
    adaptive_threshold=False,
    adapt_tau_bounds_ms=None,
    adapt_strength_init_mv=1.0,
    adapt_strength_max_mv=None,
    recording_mode="full",
    output_fields=None,
    refractory_e_ms=None,
    refractory_i_ms=None,
    refractory_policy="nearest",
):
    """Run inference on a single sample and save full spike trajectory to recording.npz.

    Sample selection: by default the trial is chosen by (digit class, index within
    class) for MNIST, or raw index for other datasets. Pass sample_index to force a
    raw test-set index regardless of dataset — notebooks that grab "test trial N"
    (rather than "digit D instance S") use this to reproduce a specific raster.
    recording_mode defaults to full traces; spikes and inhibitory retain only
    E/I or I trajectories respectively, plus the usual snapshot metadata.
    """
    import numpy as np

    # Shared per-run setup (seed, pin dt, resolve hidden_sizes — see infer()).
    _pin_run(dt, t_ms, tau_gaba=tau_gaba, seed=seed)
    hidden_sizes = _resolve_hidden_sizes(
        hidden_sizes, load_weights, [DATASET_N_HIDDEN_DEFAULTS.get(dataset, 256)]
    )
    device = _auto_device()

    # Load dataset
    _, X_te, _, y_te = load_dataset(
        dataset,
        max_samples=None,
        split=True,
        evaluation_split="test",
        evaluation_only=True,
    )
    M.N_IN = 784

    # Select single sample. A raw sample_index overrides digit-class selection for
    # any dataset; otherwise MNIST selects by (digit class, index within class) and
    # other datasets use the raw index.
    if sample_index is not None:
        sample_idx = sample_index if 0 <= sample_index < len(X_te) else 0
    elif dataset == "mnist":
        digit_mask = y_te == digit
        digit_indices = np.where(digit_mask)[0]
        if sample >= len(digit_indices):
            log.warning(
                f"Sample {sample} out of range for digit {digit} "
                f"({len(digit_indices)} available); using sample 0"
            )
            sample = 0
        sample_idx = digit_indices[sample]
    else:
        sample_idx = sample if sample < len(X_te) else 0

    X_single = torch.from_numpy(X_te[sample_idx : sample_idx + 1]).to(device)
    y_single = torch.from_numpy(y_te[sample_idx : sample_idx + 1]).to(device)

    # Build model
    net = build_net(
        model_name,
        refractory_e_ms=refractory_e_ms,
        refractory_i_ms=refractory_i_ms,
        refractory_policy=refractory_policy,
        w_in=w_in,
        w_in_initial_zero_fraction=w_in_initial_zero_fraction,
        recurrent_initial_zero_fraction=recurrent_initial_zero_fraction,
        ei_strength=ei_strength,
        ei_ratio=ei_ratio,
        device=device,
        randomize_init=True,
        dales_law=dales_law,
        hidden_sizes=hidden_sizes,
        readout_mode=readout_mode,
        signed_readout=signed_readout,
        readout_bias=readout_bias,
        train_leak=train_leak,
        tau_m_e_bounds_ms=tau_m_e_bounds_ms,
        tau_m_i_bounds_ms=tau_m_i_bounds_ms,
        adaptive_threshold=adaptive_threshold,
        adapt_tau_bounds_ms=adapt_tau_bounds_ms,
        adapt_strength_init_mv=adapt_strength_init_mv,
        adapt_strength_max_mv=adapt_strength_max_mv,
    )

    # Load weights (skip_load drops matching keys by prefix — see infer()).
    assert load_weights is not None
    state = torch.load(load_weights, map_location=device)
    if skip_load:
        state = {
            k: v
            for k, v in state.items()
            if not any(k.startswith(p) for p in skip_load)
        }
    net.load_state_dict(state, strict=False)

    # Optional hidden-spike perturbation for the snapshot (same hook as infer()).
    if perturb_mode is not None:
        _pgen = torch.Generator(device=device).manual_seed(EVAL_SEED + 1)
        net._hidden_perturb_fn = _make_perturb_fn(
            perturb_mode, perturb_level, dt, _pgen
        )

    # Optional single-trial I-spike override (B=1): substitute s_i[t] each step.
    if i_override_file is not None:
        z = np.load(i_override_file)
        T_ov, n_i_ov = int(z["T"]), int(z["n_i"])
        ov = np.zeros((T_ov, n_i_ov), dtype="float32")
        ov[z["i_t"], z["i_cell"]] = 1.0  # trial 0 only
        _ov_t = torch.from_numpy(ov).to(device)
        _st = {"step": 0}

        def _ov_fn(s_e, s_i, _layer, _ov=_ov_t, _st=_st):
            t = _st["step"]
            _st["step"] = t + 1
            if s_i is None:
                return s_e, s_i
            return s_e, _ov[t].unsqueeze(0) if s_i.ndim == 2 else _ov[t]

        net._hidden_perturb_fn = _ov_fn

    # Run forward pass with spike recording enabled
    # The selected recording mode controls which trajectory buffers are allocated.
    net.recording = True
    net.recording_mode = recording_mode
    net.eval()  # Disable dropout, batch norm, etc.
    with torch.no_grad():
        # Encode pixel data as Poisson spike train using EVAL_SEED for determinism
        spk = encode_batch(X_single, dt)
        net(input_spikes=spk)  # records spikes into net.spike_record (used below)

    # Extract and save spike recording to NPZ file for notebook analysis
    # The recording dict contains spike tensors keyed by layer ('hid', 'inh', etc.),
    # plus optional traces (voltages, conductances, etc.) if the network recorded them.
    rec = net.spike_record
    assert out_dir is not None, "infer_and_snapshot requires out_dir"
    out_path = Path(out_dir) / "recording.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save snapshot with canonical field names and metadata (including the sample's
    # true class label, for rasters that annotate the digit).
    # (Uses shared utility to ensure all snapshot fields are consistent across code paths)
    save_snapshot_npz(
        out_path,
        rec,
        dt,
        M.N_HID,
        M.N_INH,
        label=int(y_single.item()),
        output_fields=output_fields,
    )

    return {"acc": None}


def probe(
    model_name="ping",
    dt=0.1,
    t_ms=200.0,
    hidden_sizes=None,
    n_in=784,
    n_inh=None,
    ei_strength=1.0,
    ei_ratio=2.0,
    w_ei_mean=None,
    w_ie_mean=None,
    w_in=None,
    w_in_initial_zero_fraction=0.0,
    recurrent_initial_zero_fraction=0.0,
    dales_law=True,
    seed=None,
    load_weights=None,
    input_rate_hz=25.0,
    n_batch=64,
    input_file=None,
    out_dir=None,
    outputs=None,
    tau_gaba=None,
    private_w_in=False,
    train_leak=False,
    tau_m_e_bounds_ms=None,
    tau_m_i_bounds_ms=None,
    adaptive_threshold=False,
    adapt_tau_bounds_ms=None,
    adapt_strength_init_mv=1.0,
    adapt_strength_max_mv=None,
    readout_mode="rate",
    signed_readout=False,
    readout_bias=False,
    scale_w_in=1.0,
    scale_w_ei=1.0,
    scale_w_ie=1.0,
    output_fields=None,
    recording_mode="full",
    recording_start_step=0,
    refractory_e_ms=None,
    refractory_i_ms=None,
    refractory_policy="nearest",
):
    """Drive a net with uniform homogeneous Poisson input; emit E/I rates.

    Builds a network (untrained if load_weights is None, else loaded) with the
    given recurrent structure, drives it with a uniform Poisson spike train
    (every input channel independent at input_rate_hz) for n_batch trials, and
    writes population E/I firing rates to metrics.json. Optional --outputs add
    rasters.npz / per_cell_rates.npz for cycle- or cell-level analysis.

    recording_start_step trims only emitted sparse events; dynamics and reported
    rates use the complete input window. Retained times remain absolute indices.
    No dataset and no accuracy — this is the untrained-net / f-I-curve probe path.
    ei_strength/ei_ratio set the recurrent means (w_ei=(s, s·0.1),
    w_ie=(s·ratio, s·ratio·0.1)); n_inh sets the I-pool size. private_w_in wires
    each E cell to its own input channel (identity W_in) for input-decorrelated
    probes.
    """
    import numpy as np

    # Shared per-run setup (see infer()); probe pins N_IN explicitly and can
    # override N_INH after the hidden sizes are installed.
    _pin_run(dt, t_ms, tau_gaba=tau_gaba, seed=seed)
    M.N_IN = int(n_in)
    hidden_sizes = _resolve_hidden_sizes(hidden_sizes, load_weights, [256])
    if n_inh is not None:
        M.N_INH = int(n_inh)

    device = _auto_device()
    n_inh_per_layer = {1: int(n_inh)} if n_inh is not None else None
    # Independent recurrent means (w_ei_mean/w_ie_mean) override ei_strength/ei_ratio
    # when given — needed for 2D (W_EI, W_IE) plane sweeps where either can be zero.
    # std held at 10% of the mean, matching the notebook convention.
    build_kwargs = dict(
        refractory_e_ms=refractory_e_ms,
        refractory_i_ms=refractory_i_ms,
        refractory_policy=refractory_policy,
        w_in=w_in,
        w_in_initial_zero_fraction=w_in_initial_zero_fraction,
        recurrent_initial_zero_fraction=recurrent_initial_zero_fraction,
        device=device,
        randomize_init=True,
        dales_law=dales_law,
        hidden_sizes=hidden_sizes,
        readout_mode=readout_mode,
        signed_readout=signed_readout,
        readout_bias=readout_bias,
        n_inh_per_layer=n_inh_per_layer,
        train_leak=train_leak,
        tau_m_e_bounds_ms=tau_m_e_bounds_ms,
        tau_m_i_bounds_ms=tau_m_i_bounds_ms,
        adaptive_threshold=adaptive_threshold,
        adapt_tau_bounds_ms=adapt_tau_bounds_ms,
        adapt_strength_init_mv=adapt_strength_init_mv,
        adapt_strength_max_mv=adapt_strength_max_mv,
    )
    if w_ei_mean is not None or w_ie_mean is not None:
        build_kwargs["w_ei"] = (float(w_ei_mean or 0.0), float(w_ei_mean or 0.0) * 0.1)
        build_kwargs["w_ie"] = (float(w_ie_mean or 0.0), float(w_ie_mean or 0.0) * 0.1)
    else:
        build_kwargs["ei_strength"] = ei_strength
        build_kwargs["ei_ratio"] = ei_ratio
    net = build_net(model_name, **build_kwargs)
    if private_w_in:
        # One input channel per E cell (identity W_in) — removes shared-input
        # coincidences so the measured rhythmicity is input-decorrelated.
        import torch as _t

        n_e = hidden_sizes[-1]
        with _t.no_grad():
            w = net.W_ff[0]
            scale = float(w_in[0]) if w_in else 1.0
            net.W_ff[0].copy_(_t.eye(n_e, device=w.device) * scale)
    if load_weights is not None:
        state = torch.load(load_weights, map_location=device)
        net.load_state_dict(state, strict=False)
    if scale_w_in != 1.0:
        net.W_ff[0].data.mul_(float(scale_w_in))
    if scale_w_ei != 1.0:
        for key in net.W_ei:
            net.W_ei[key].data.mul_(float(scale_w_ei))
    if scale_w_ie != 1.0:
        for key in net.W_ie:
            net.W_ie[key].data.mul_(float(scale_w_ie))
    if (scale_w_in, scale_w_ei, scale_w_ie) != (1.0, 1.0, 1.0):
        log.info(
            f"  scaled weights: W_in×{scale_w_in} W_ei×{scale_w_ei} W_ie×{scale_w_ie}"
        )
    net.eval()
    if recording_mode not in ("full", "spikes"):
        raise ValueError("probe recording requires E/I spikes")
    if "spike_summary" in (outputs or ()) and recording_mode != "full":
        raise ValueError("spike_summary requires output spikes from full recording")
    net.recording_mode = recording_mode
    net.recording = True

    outputs = set(outputs or ())
    emit_rasters = "rasters" in outputs
    emit_spike_summary = "spike_summary" in outputs
    emit_per_cell = "per_cell_rates" in outputs

    # Input: either an arbitrary pre-built spike stream (--input-file, the generic
    # "run the net on this exact input" primitive — dual of --outputs rasters) or a
    # single uniform-Poisson draw of shape (T, n_batch, N_IN) at input_rate_hz.
    if input_file is not None:
        loaded_input = np.load(input_file)
        arr = loaded_input["input_spikes"]
        readout_reset = (
            torch.from_numpy(loaded_input["readout_reset"]).to(device)
            if "readout_reset" in loaded_input.files
            else None
        )
        spk_in = torch.from_numpy(arr).float()
        if spk_in.ndim == 2:  # (T, N_IN) → (T, 1, N_IN)
            spk_in = spk_in.unsqueeze(1)
        spk_in = spk_in.to(device)
        T_steps, n_batch = spk_in.shape[0], spk_in.shape[1]
        M.T_steps = T_steps
        # Preserve the nominal request pinned by _pin_run; the supplied input
        # determines actual steps and the model records both durations.
        log.info(f"  input-file: {input_file}  shape={tuple(spk_in.shape)}")
    else:
        T_steps = duration_steps(t_ms, dt)
        p_step = input_rate_hz * dt / 1000.0
        gen = torch.Generator().manual_seed((seed or 0) + 1)
        spk_in = (
            (torch.rand(T_steps, int(n_batch), M.N_IN, generator=gen) < p_step)
            .float()
            .to(device)
        )
        readout_reset = None
    if not 0 <= recording_start_step < T_steps:
        raise ValueError("recording start must lie inside the simulation")
    with torch.no_grad():
        net(input_spikes=spk_in, readout_reset_mask=readout_reset)

    rec = net.spike_record
    hk, ik = primary_hid_key(rec), primary_inh_key(rec)
    t_sec = T_steps * dt / 1000.0
    n_e = M.N_HID
    n_i = M.N_INH or 1
    e_sum = float(rec[hk].sum().item()) if hk else 0.0
    i_sum = float(rec[ik].sum().item()) if ik else 0.0
    r_e = e_sum / (n_batch * n_e * t_sec)
    r_i = i_sum / (n_batch * n_i * t_sec) if ik else 0.0
    log.info(
        f"  probe: rate_e={r_e:.2f}Hz rate_i={r_i:.2f}Hz (n_batch={n_batch}, input={input_rate_hz}Hz)"
    )

    out_dir_path = Path(out_dir) if out_dir else None
    if out_dir_path and out_dir_path.exists():
        blob = {
            "mode": "probe",
            "model": model_name,
            "config": {
                "dt": dt,
                "t_ms": t_ms,
                **duration_metadata(t_ms, dt),
                "duration_steps": T_steps,
                "realized_duration_ms": T_steps * dt,
                **refractory_metadata(
                    net.refractory_e_ms, net.refractory_i_ms, dt,
                    policy=net.refractory_policy,
                ),
                "n_in": int(n_in),
                "n_hidden": n_e,
                "n_inh": n_i,
                "ei_strength": ei_strength,
                "ei_ratio": ei_ratio,
                "input_rate_hz": input_rate_hz,
                "n_batch": int(n_batch),
                "load_weights": str(load_weights) if load_weights else None,
            },
            "rate_e_hz": r_e,
            "rate_i_hz": r_i,
            "rates_hz": {"hid": r_e, "inh": r_i},
        }
        with open(out_dir_path / "metrics.json", "w") as f:
            json.dump(blob, f, indent=2, default=float)
        log.info(f"  → {out_dir_path / 'metrics.json'}")

        if emit_per_cell:
            per_e = (
                rec[hk].sum(dim=(0, 1)).detach().cpu().numpy() if hk else np.zeros(0)
            )
            dump = {"rate_e_per_cell": (per_e / (n_batch * t_sec)).astype(np.float32)}
            if ik:
                per_i = rec[ik].sum(dim=(0, 1)).detach().cpu().numpy()
                dump["rate_i_per_cell"] = (per_i / (n_batch * t_sec)).astype(np.float32)
            save_selected_npz(out_dir_path / "per_cell_rates.npz", dump, output_fields)

        if emit_rasters:

            def _coo(key):
                r = rec.get(key)
                if r is None:
                    return (np.zeros(0, np.int32),) * 3
                a = r.detach().cpu().numpy()
                if a.ndim == 2:
                    a = a[:, None, :]
                tt, bb, cc = a[recording_start_step:].nonzero()
                tt = tt + recording_start_step
                return bb.astype("int32"), tt.astype("int32"), cc.astype("int32")

            e_tr, e_t, e_c = _coo(hk)
            i_tr, i_t, i_c = _coo(ik)
            out_tr, out_t, out_c = (
                _coo("out_spikes")
                if output_fields is None or "out_t" in output_fields
                else (np.zeros(0, np.int32),) * 3
            )
            save_selected_npz(
                out_dir_path / "rasters.npz",
                dict(
                    dt=np.float32(dt),
                    n_trials=np.int32(n_batch),
                    T=np.int32(T_steps),
                    n_e=np.int32(n_e),
                    n_i=np.int32(n_i),
                    e_trial=e_tr,
                    e_t=e_t,
                    e_cell=e_c,
                    i_trial=i_tr,
                    i_t=i_t,
                    i_cell=i_c,
                    out_trial=out_tr,
                    out_t=out_t,
                    out_cell=out_c,
                    **(
                        {"recording_start_step": np.int32(recording_start_step)}
                        if recording_start_step
                        else {}
                    ),
                ),
                output_fields,
            )
            log.info(
                f"  → {out_dir_path / 'rasters.npz'}  "
                f"({e_tr.size + i_tr.size + out_tr.size} spikes)"
            )

        if emit_spike_summary:

            def _as_tbc(key, width):
                value = rec.get(key)
                if value is None:
                    return torch.zeros((T_steps, n_batch, width), device=device)
                if value.ndim == 2:
                    value = value.unsqueeze(1)
                return value

            starts = [0]
            if readout_reset is not None:
                reset_indices = (
                    torch.nonzero(readout_reset.detach().bool().cpu(), as_tuple=False)
                    .flatten()
                    .tolist()
                )
                starts.extend(int(index) for index in reset_indices if int(index) > 0)
            starts = sorted(set(starts))
            stops = [*starts[1:], T_steps]

            e_spikes = _as_tbc(hk, n_e)
            i_spikes = _as_tbc(ik, n_i)
            out_spikes = _as_tbc("out_spikes", M.N_OUT)

            def _population_counts(value):
                return (
                    torch.stack(
                        [
                            value[start:stop].sum(dim=(0, 2))
                            for start, stop in zip(starts, stops)
                        ]
                    )
                    .transpose(0, 1)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.int64)
                )

            output_counts = (
                torch.stack(
                    [
                        out_spikes[start:stop].sum(dim=0)
                        for start, stop in zip(starts, stops)
                    ]
                )
                .permute(1, 0, 2)
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64)
            )
            np.savez_compressed(
                out_dir_path / "spike_summary.npz",
                dt=np.float32(dt),
                n_trials=np.int32(n_batch),
                T=np.int32(T_steps),
                segment_starts=np.asarray(starts, dtype=np.int32),
                segment_stops=np.asarray(stops, dtype=np.int32),
                e_counts=_population_counts(e_spikes),
                i_counts=_population_counts(i_spikes),
                out_counts=output_counts,
            )
            log.info(f"  → {out_dir_path / 'spike_summary.npz'}")

    return {"rate_e_hz": r_e, "rate_i_hz": r_i}


def dump_weights(
    model_name="ping",
    dt=0.25,
    load_weights=None,
    dataset="mnist",
    t_ms=200.0,
    w_in=None,
    ei_strength=0.5,
    ei_ratio=2.0,
    w_in_initial_zero_fraction=0.0,
    recurrent_initial_zero_fraction=0.0,
    hidden_sizes=None,
    out_dir=None,
    dales_law=True,
    seed=None,
    readout_mode="rate",
    signed_readout=False,
    readout_bias=False,
    trainable_w_ee=False,
    trainable_w_ei=False,
    trainable_w_ie=False,
    trainable_w_ii=False,
    kaiming_init=False,
    train_leak=False,
    tau_m_e_bounds_ms=None,
    tau_m_i_bounds_ms=None,
    adaptive_threshold=False,
    adapt_tau_bounds_ms=None,
    adapt_strength_init_mv=1.0,
    adapt_strength_max_mv=None,
    output_fields=None,
    refractory_e_ms=None,
    refractory_i_ms=None,
    refractory_policy="nearest",
):
    """Emit initialisation and trained weight matrices to weights_dump.npz.

    Rebuilds the network under its training seed to recover the deterministic
    INIT weights (build_net with randomize_init is seed-reproducible), then reads
    the trained values straight out of the saved state_dict. Lets notebooks
    compare init-vs-trained anatomical weights without importing build_net.

    For every fixed E-I weight matrix (W_ei, W_ie, W_ee, W_ii) and layer key k,
    emits two arrays: <name>_<k>_init and <name>_<k>_trained. E.g. a single
    E-I-layer PING gives W_ei_1_init, W_ei_1_trained, W_ie_1_init, W_ie_1_trained.

    Args mirror infer() so the CLI maps --load-config → this the same way.
    """
    import numpy as np

    # Seed BEFORE build_net so the randomized init is byte-identical to training's
    # init (this is the whole point — reproduce the pre-training weights). Resolve
    # hidden_sizes from the checkpoint too, so the init arrays match the trained
    # shapes for a non-default-size net.
    _pin_run(dt, t_ms, seed=seed)
    hidden_sizes = _resolve_hidden_sizes(
        hidden_sizes, load_weights, [DATASET_N_HIDDEN_DEFAULTS.get(dataset, 256)]
    )
    M.N_IN = 784

    device = _auto_device()
    net = build_net(
        model_name,
        refractory_e_ms=refractory_e_ms,
        refractory_i_ms=refractory_i_ms,
        refractory_policy=refractory_policy,
        w_in=w_in,
        w_in_initial_zero_fraction=w_in_initial_zero_fraction,
        recurrent_initial_zero_fraction=recurrent_initial_zero_fraction,
        ei_strength=ei_strength,
        ei_ratio=ei_ratio,
        device=device,
        # kaiming cells are already heterogeneous; everything else needs the
        # seeded symmetry-break, exactly as train/infer do.
        randomize_init=not kaiming_init,
        dales_law=dales_law,
        hidden_sizes=hidden_sizes,
        readout_mode=readout_mode,
        signed_readout=signed_readout,
        readout_bias=readout_bias,
        trainable_w_ee=trainable_w_ee,
        trainable_w_ei=trainable_w_ei,
        trainable_w_ie=trainable_w_ie,
        trainable_w_ii=trainable_w_ii,
        train_leak=train_leak,
        tau_m_e_bounds_ms=tau_m_e_bounds_ms,
        tau_m_i_bounds_ms=tau_m_i_bounds_ms,
        adaptive_threshold=adaptive_threshold,
        adapt_tau_bounds_ms=adapt_tau_bounds_ms,
        adapt_strength_init_mv=adapt_strength_init_mv,
        adapt_strength_max_mv=adapt_strength_max_mv,
    )

    _WEIGHT_DICTS = ("W_ei", "W_ie", "W_ee", "W_ii")

    # INIT weights: read straight off the freshly built (untrained) net.
    dump: dict[str, "np.ndarray"] = {}

    def retain(name, value):
        if output_fields is None or name in output_fields:
            dump[name] = (
                value.detach().cpu().numpy()
                if hasattr(value, "detach")
                else np.asarray(value)
            )

    for name in _WEIGHT_DICTS:
        pdict = getattr(net, name, None)
        if pdict is None:
            continue
        for k, w in pdict.items():
            retain(f"{name}_{k}_init", w)

    # Feed-forward weights (W_ff ParameterList): W_ff[0] = W_in, W_ff[-1] = W_out.
    # Emitted as W_ff_<i>_init so notebooks can read the readout matrix (W_out).
    if hasattr(net, "W_ff"):
        for i, w in enumerate(net.W_ff):
            retain(f"W_ff_{i}_init", w)
    for name in (
        "b_out",
        "readout_alpha",
        "tau_m_e_logit",
        "tau_m_i_logit",
        "adapt_tau_logit",
        "adapt_strength_logit",
    ):
        parameter = getattr(net, name, None)
        if parameter is not None:
            if isinstance(parameter, torch.nn.ParameterDict):
                for k, value in parameter.items():
                    retain(f"{name}_{k}_init", value)
            else:
                retain(f"{name}_init", parameter)

    # TRAINED weights: pulled from the saved state_dict (keys look like "W_ei.1"
    # or "W_ff.2"). W_ff entries are emitted too; the last index is W_out.
    assert load_weights is not None, "dump_weights requires --load-weights"
    state = torch.load(load_weights, map_location="cpu")
    for sk, sv in state.items():
        name, _, key = sk.partition(".")
        if name in _WEIGHT_DICTS and key:
            retain(f"{name}_{key}_trained", sv)
        elif name == "W_ff" and key:
            retain(f"W_ff_{key}_trained", sv)
        elif sk in ("b_out", "readout_alpha"):
            retain(f"{sk}_trained", sv)
        elif (
            name
            in (
                "tau_m_e_logit",
                "tau_m_i_logit",
                "adapt_tau_logit",
                "adapt_strength_logit",
            )
            and key
        ):
            retain(f"{name}_{key}_trained", sv)

    assert out_dir is not None, "dump_weights requires out_dir"
    out_path = Path(out_dir) / "weights_dump.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_selected_npz(out_path, dump, output_fields)
    log.info(f"  → {out_path}  ({len(dump)} arrays)")

    return {"n_arrays": len(dump), "path": str(out_path)}
