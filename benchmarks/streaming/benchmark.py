"""Measure streaming deadlines and processing phases, with a two-panel summary."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

from snnlab import lang
from snnlab.sim import (
    GraphExecutor,
    Performance,
    StreamSession,
    plan_graph,
    prepare_stream,
)

STATE_GROUPS = (
    "voltages",
    "refractory",
    "conductances",
    "currents",
    "population_histories",
    "input_histories",
    "custom_state",
)


@dataclass(frozen=True)
class Config:
    hidden: int = 128
    inputs: int = 64
    readouts: int = 8
    sparsity: float = 0.9
    rate_hz: float = 25.0
    dt_ms: float = 0.1
    window_ms: float = 1.0
    period_ms: float = 1.0
    spin_us: float = 1000.0
    windows: int = 256
    seed: int = 7
    device: str = "cpu"
    compile_mode: str = "default"
    block_steps: int = 10
    warmup_windows: int = 20
    threads: int = 1

    def validate(self):
        for name in ("hidden", "inputs", "readouts", "windows", "threads"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("dt_ms", "window_ms", "period_ms"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if round(self.period_ms * 1e6) < 1:
            raise ValueError("period_ms must resolve to at least one nanosecond")
        if not math.isfinite(self.spin_us) or self.spin_us < 0:
            raise ValueError("spin_us must be finite and nonnegative")
        ratio = self.window_ms / self.dt_ms
        if ratio < 1 or not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-6):
            raise ValueError("window / timestep must be a positive integer")
        if not math.isfinite(self.rate_hz) or self.rate_hz < 0:
            raise ValueError("rate_hz must be nonnegative and finite")
        if not math.isfinite(self.sparsity) or not 0 <= self.sparsity < 1:
            raise ValueError("sparsity must be in [0, 1)")
        if not 1 <= self.block_steps <= 64 or self.warmup_windows < 2:
            raise ValueError("block_steps must be in [1,64]; warmup_windows >= 2")
        if self.device not in ("cpu", "cuda", "mps"):
            raise ValueError("device must be cpu, cuda or mps")
        if self.compile_mode not in ("default", "eager"):
            raise ValueError("compile_mode must be default or eager")

    @property
    def steps(self):
        return round(self.window_ms / self.dt_ms)


def circuit(cfg):
    net = lang.Network("streaming-random", dt=cfg.dt_ms * lang.ms)
    events = net.input(
        "events",
        shape=("time", "batch", cfg.inputs),
        signal_type="spikes",
        unit="spike",
    )
    hidden = net.population(
        "hidden",
        size=cfg.hidden,
        neuron=lang.COBA_LIF(tau_mem=20 * lang.ms, refractory=2 * lang.ms),
    )
    readout = net.population(
        "readout",
        size=cfg.readouts,
        neuron=lang.COBA_LIF(tau_mem=20 * lang.ms, refractory=2 * lang.ms),
    )
    for name, source, target, strength, recurrent in (
        ("input_hidden", events, hidden, 0.08, False),
        ("hidden_hidden", hidden.spikes, hidden, 0.02, True),
        ("hidden_readout", hidden.spikes, readout, 0.10, False),
    ):
        net.connect(
            source,
            target.excitatory,
            name=name,
            synapse=lang.AMPA(tau=5 * lang.ms),
            weight=lang.LowerClampedNormal(
                strength, strength / 5, initial_zero_fraction=cfg.sparsity
            ),
            initialization_scaling="fan_in_normalized",
            connection="recurrent" if recurrent else "feedforward",
            delay=cfg.dt_ms * lang.ms if recurrent else None,
            allow_autapses=False,
        )
    net.output("hidden_spikes", hidden.spikes)
    net.output("readout_spikes", readout.spikes)
    return lang.compile(net).graph


def executor(cfg, plan, *, eager=False):
    return (
        GraphExecutor(
            plan,
            seed=cfg.seed,
            compile_mode=None if eager or cfg.compile_mode == "eager" else "default",
            compile_block_steps=cfg.block_steps,
            precision="highest",
            cuda_graphs="off",
        )
        .to(cfg.device)
        .eval()
    )


def stream(cfg, plan):
    return prepare_stream(
        plan,
        window_steps=cfg.steps,
        batch_size=1,
        seed=cfg.seed,
        device=cfg.device,
        performance=Performance(
            compile_mode=None if cfg.compile_mode == "eager" else "default",
            compile_block_steps=cfg.block_steps,
            precision="highest",
            cuda_graphs="off",
        ),
    )


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def input_window(cfg, rng):
    # Poisson counts per 0.1 ms bin, generated on the host inside callback timing.
    return torch.from_numpy(
        rng.poisson(cfg.rate_hz * cfg.dt_ms / 1000, (cfg.steps, 1, cfg.inputs)).astype(
            np.float32
        )
    ).to(cfg.device)


def advance(model, values, state=None):
    if isinstance(model, StreamSession):
        if state is not None:
            raise ValueError("prepared streams own their state; use reset(snapshot)")
        return model.advance({"events": values})
    return model(
        {"events": values},
        runtime_state=state,
        diagnostics=False,
        retain_parameters=False,
        retain_final_state=False,
    )


def warm_and_validate(cfg, model, plan):
    """Stress boundary-spanning dynamics; compare eager whole vs measured chunks."""
    started = time.perf_counter()
    reference = executor(cfg, plan, eager=True)
    model.reset()
    rng = np.random.default_rng(cfg.seed + 1)
    windows = [input_window(cfg, rng) for _ in range(cfg.warmup_windows)]
    # Synchronous probe bursts guarantee substantial drive even at low input rates.
    for values in windows:
        values[0] = 1
    synchronize(cfg.device)
    cold_started = time.perf_counter()
    first = advance(model, windows[0])
    synchronize(cfg.device)
    cold_s = time.perf_counter() - cold_started
    results = [first]
    for values in windows[1:]:
        results.append(advance(model, values))
    whole = advance(reference, torch.cat(windows))
    for name, expected in whole.outputs.items():
        torch.testing.assert_close(
            torch.cat([r.outputs[name] for r in results]), expected, atol=0, rtol=0
        )
    stream_state = model.export_state()
    for group in STATE_GROUPS:
        expected = getattr(whole.runtime_state, group)
        actual = getattr(stream_state, group)
        assert expected.keys() == actual.keys(), group
        for name in expected:
            torch.testing.assert_close(
                actual[name], expected[name], atol=3e-4, rtol=3e-4
            )
    assert whole.runtime_state.completed_steps == model.completed_steps
    paths = [r.metrics["execution_path"] for r in results]
    synchronize(cfg.device)
    model.reset()
    return {
        "passed": True,
        "windows": len(windows),
        "steps": len(windows) * cfg.steps,
        "probe": "seeded Poisson input with synchronous burst at every window start",
        "spikes_exact": True,
        "state_atol": 3e-4,
        "state_rtol": 3e-4,
        "first_prepared_window_s": cold_s,
        "validation_and_warmup_s": time.perf_counter() - started,
        "initial_execution_path": paths[0],
        "continuation_execution_path": paths[-1],
        "probe_spikes": {
            name: int(value.sum().item()) for name, value in whole.outputs.items()
        },
    }


def extract_output(values, result):
    # Materialize all readout events and the same activity counters on the host.
    events = result.outputs["readout_spikes"].nonzero().cpu().tolist()
    return (
        int(result.outputs["hidden_spikes"].sum().item()),
        len(events),
        int(values.sum().item()),
    )


BREAKDOWN_STAGES = (
    "Full callback",
    "Input preparation",
    "Prepared stream advance",
    "Output extraction",
    "Other / measurement",
)


PHASES = (
    "Input validation",
    "Input buffer copy",
    "Executor entry / validation",
    "State restore / init",
    "Execution setup / metadata",
    "Compiled numerical block",
    "Eager simulation / histories",
    "Output / history assembly",
    "Runtime-state snapshot",
    "Executor result / metadata",
    "Output copies",
    "Session result / commit",
)


class PhaseProbe:
    def __init__(self, synchronize):
        self.synchronize = synchronize
        self.indices = {name: index for index, name in enumerate(PHASES)}

    @staticmethod
    def stamp():
        return np.array((time.perf_counter_ns(), time.thread_time_ns()), dtype=np.int64)

    def begin(self):
        self.total = np.zeros((len(PHASES), 2), dtype=np.int64)
        self.current = None
        self.full_start = self.stamp()

    def mark(self, name):
        # Completion wait is charged to the preceding phase, not the following one.
        self.synchronize()
        stop = self.stamp()
        if self.current is not None:
            self.total[self.indices[self.current]] += stop - self.started
        self.current = name
        # Time spent updating the probe remains in the explicit remainder.
        self.started = self.stamp()

    def finish(self):
        self.mark(None)
        full = self.stamp() - self.full_start
        remainder = full - self.total.sum(axis=0)
        if (remainder < 0).any():
            raise RuntimeError("phase intervals exceeded the full advance interval")
        return np.vstack((full, self.total, remainder))


@torch.inference_mode()
def measure_advance_phases(cfg, session, windows):
    rng = np.random.default_rng(cfg.seed + 3)
    inputs = [input_window(cfg, rng) for _ in range(windows)]

    def warm():
        session.reset()
        advance(session, inputs[0])
        return advance(session, inputs[-1])

    warm()
    expected = [advance(session, values).outputs for values in inputs]
    expected_state = session.export_state()
    warm_result = warm()
    synchronize(cfg.device)
    probe = PhaseProbe(lambda: synchronize(cfg.device))
    executor = session._executor
    old_session = session._advance_probe
    old_executor = getattr(executor, "_stream_probe", None)
    data = np.empty((windows, len(PHASES) + 2, 2), dtype=np.int64)
    paths = set()
    actual = []
    try:
        session._advance_probe = probe
        executor._stream_probe = probe
        for index, values in enumerate(inputs):
            probe.begin()
            result = advance(session, values)
            data[index] = probe.finish()
            actual.append(result.outputs)
            paths.add(result.metrics["execution_path"]["selected"])
    finally:
        session._advance_probe = old_session
        executor._stream_probe = old_executor
    expected_path = warm_result.metrics["execution_path"]["selected"]
    if paths != {expected_path}:
        raise RuntimeError(f"Phase instrumentation changed execution path: {paths}")
    if session.completed_steps != (windows + 2) * cfg.steps:
        raise RuntimeError("phase capture advanced an unexpected number of steps")
    for reference, observed in zip(expected, actual, strict=True):
        for name, value in reference.items():
            torch.testing.assert_close(observed[name], value, atol=0, rtol=0)
    observed_state = session.export_state()
    for group in STATE_GROUPS:
        for name, value in getattr(expected_state, group).items():
            torch.testing.assert_close(
                getattr(observed_state, group)[name], value, atol=0, rtol=0
            )
    names = ("Full advance", *PHASES, "Other / measurement")
    return {
        "windows": windows,
        "execution_path": warm_result.metrics["execution_path"],
        "numerical_gate_passed": True,
        "statistic": "arithmetic mean per prepared advance",
        "scope": "Separate warmed unpaced pass with phase-boundary wall/thread-CPU clocks and device completion fences; no PyTorch profiler or cProfile active.",
        "limits": "Phases are nonoverlapping. Full advance is inclusive. Other includes timer bookkeeping and API entry/exit. Compiled block includes dispatch, input views, invocation and completion wait, not just numerical kernels. Thread CPU excludes workers/device compute. Instrumentation changes timing.",
        "stages": [
            {
                "name": name,
                "wall_us": float(data[:, index, 0].mean() / 1000),
                "thread_cpu_us": float(data[:, index, 1].mean() / 1000),
            }
            for index, name in enumerate(names)
            if index in (0, len(names) - 1) or data[:, index].any()
        ],
    }


def measure_breakdown(cfg, model):
    """Unpaced callback timing, separate from the internal phase capture."""
    rows = np.empty((cfg.windows, len(BREAKDOWN_STAGES), 2), dtype=np.int64)
    rng = np.random.default_rng(cfg.seed + 2)
    model.reset()

    def stamp():
        return np.array((time.perf_counter_ns(), time.thread_time_ns()), dtype=np.int64)

    for index in range(cfg.windows):
        start = stamp()
        values = input_window(cfg, rng)
        synchronize(cfg.device)
        prepared = stamp()
        result = advance(model, values)
        synchronize(cfg.device)
        executed = stamp()
        extract_output(values, result)
        synchronize(cfg.device)
        extracted = stamp()
        rows[index, 0] = extracted - start
        rows[index, 1] = prepared - start
        rows[index, 2] = executed - prepared
        rows[index, 3] = extracted - executed
        rows[index, 4] = rows[index, 0] - rows[index, 1:4].sum(axis=0)

    return {
        "windows": cfg.windows,
        "stages": [
            {
                "name": name,
                "wall_us": float(rows[:, i, 0].mean() / 1000),
                "thread_cpu_us": float(rows[:, i, 1].mean() / 1000),
            }
            for i, name in enumerate(BREAKDOWN_STAGES)
        ],
    }


def deadline_schedule(
    callback, windows, period_ns, *, spin_ns=0, clock=None, sleep=None
):
    """Absolute monotonic releases; catch up without dropping or shifting windows.

    Columns contain scheduled release, callback start and completed callback end,
    in nanoseconds relative to one epoch. Array writes happen after callback end;
    their overhead, like waiting/scheduling overhead, can delay the next start.
    Clock/sleep injection lets tests exercise overruns without wall-clock timing.
    """
    clock = time.perf_counter_ns if clock is None else clock
    sleep = time.sleep if sleep is None else sleep
    samples = np.empty((windows, 3), dtype=np.int64)
    epoch = clock()
    for index in range(windows):
        release = epoch + index * period_ns
        while True:
            now = clock()
            if now >= release:
                break
            remaining = release - now
            if remaining > spin_ns:
                sleep((remaining - spin_ns) / 1e9)
            # The final guard is spent polling the same monotonic clock. No
            # early releases, shifted deadlines or skipped model windows.
        started = clock()
        callback()
        finished = clock()
        samples[index] = (release - epoch, started - epoch, finished - epoch)
    return samples


def deadline_arrays(samples, period_ns):
    release, started, finished = samples.T
    return {
        "release_ns": release,
        "start_ns": started,
        "finish_ns": finished,
        "deadline_ns": release + period_ns,
        "callback_ns": finished - started,
        "response_ns": finished - release,
        "start_lag_ns": started - release,
        "lateness_ns": finished - (release + period_ns),
        "deadline_missed": finished > release + period_ns,
        "callback_overrun": finished - started > period_ns,
    }


def measure_deadlines(cfg, model):
    """Paced end-to-end windows, without internal phase instrumentation."""
    rng = np.random.default_rng(cfg.seed + 4)

    def callback():
        values = input_window(cfg, rng)
        result = advance(model, values)
        extract_output(values, result)
        synchronize(cfg.device)

    model.reset()
    for _ in range(cfg.warmup_windows):
        callback()
    synchronize(cfg.device)
    model.reset()
    rng = np.random.default_rng(cfg.seed + 4)
    period_ns = round(cfg.period_ms * 1e6)
    wall_started = time.perf_counter_ns()
    cpu_started = time.thread_time_ns()
    samples = deadline_schedule(
        callback, cfg.windows, period_ns, spin_ns=round(cfg.spin_us * 1000)
    )
    cpu_ns = time.thread_time_ns() - cpu_started
    wall_ns = time.perf_counter_ns() - wall_started
    arrays = deadline_arrays(samples, period_ns)
    if model.completed_steps != cfg.windows * cfg.steps:
        raise RuntimeError("deadline pass dropped or added model steps")

    def stats(key):
        ms = arrays[key] / 1e6
        return {
            "median": float(np.median(ms)),
            "p95": float(np.percentile(ms, 95)),
            "maximum": float(ms.max()),
        }

    return {
        "windows": cfg.windows,
        "period_ms": period_ns / 1e6,
        "spin_us": cfg.spin_us,
        "caller_thread_cpu_percent": 100 * cpu_ns / wall_ns,
        "paced_elapsed_ms": wall_ns / 1e6,
        "sample_columns": ["release_ns", "start_ns", "finish_ns"],
        "samples": samples.tolist(),
        "missed_deadlines": int(arrays["deadline_missed"].sum()),
        "miss_rate": float(arrays["deadline_missed"].mean()),
        "callback_budget_overruns": int(arrays["callback_overrun"].sum()),
        "callback_ms": stats("callback_ns"),
        "response_ms": stats("response_ns"),
        "start_lag_ms": stats("start_lag_ns"),
        "lateness_ms": stats("lateness_ns"),
        "completed_steps": model.completed_steps,
        "scope": "Separate warmed paced pass: input generation/transfer, stream.advance, host event extraction, and device completion. One monotonic clock for releases, starts and completion.",
        "schedule": "Absolute releases spaced by period_ms; each deadline is release + period_ms. Sleep until spin_us before release, then poll the clock. Late windows catch up without dropping model steps or shifting the epoch.",
        "limits": "Observed soft-real-time timing on this host, not a hard-deadline guarantee. Sleep and OS scheduling can cause misses even when callback duration is below budget. Timing/storage overhead can delay subsequent starts. No ROS, external sensor or actuator I/O.",
    }


@torch.inference_mode()
def run(cfg):
    cfg.validate()
    torch.set_num_threads(cfg.threads)
    torch.set_float32_matmul_precision("highest")
    plan = plan_graph(circuit(cfg))
    with stream(cfg, plan) as model:
        validation = warm_and_validate(cfg, model, plan)
        deadlines = measure_deadlines(cfg, model)
        breakdown = measure_breakdown(cfg, model)
        internal = measure_advance_phases(cfg, model, cfg.windows)
        return {
            "config": asdict(cfg),
            "validation": validation,
            "compiler_wrapper": model.info["compiler_wrapper"],
            "deadlines": deadlines,
            "breakdown": breakdown,
            "advance_breakdown": internal,
        }


def deadline_rates(release_ns, missed, overrun, interval_ms=100):
    """Rates in fixed release-time intervals; partial bins use actual samples.

    Empty intervals are missing data, not zero misses. The final interval is
    labelled with its observed window count when plotted.
    """
    interval_ns = round(interval_ms * 1e6)
    bins = np.asarray(release_ns) // interval_ns
    count = np.bincount(bins)
    missed_count = np.bincount(bins, weights=missed, minlength=len(count))
    overrun_count = np.bincount(bins, weights=overrun, minlength=len(count))
    return {
        "edges_ms": np.arange(len(count) + 1) * interval_ms,
        "windows": count,
        "miss_pct": np.divide(
            100 * missed_count, count, out=np.full(len(count), np.nan), where=count > 0
        ),
        "overrun_pct": np.divide(
            100 * overrun_count, count, out=np.full(len(count), np.nan), where=count > 0
        ),
    }


def mean_sd(values):
    """Across-seed sample SD (ddof=1); single-run error bars are omitted."""
    values = np.asarray(values, dtype=float)
    mean = values.mean(axis=0)
    sd = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros_like(mean)
    return mean, sd


def summarize_seeds(runs):
    """Equal-weight seed summaries; compute misses before averaging timings."""
    summary = {
        "seed_count": len(runs),
        "sd_defined": len(runs) > 1,
        "validation_passed": all(
            r.get("validation", {}).get("passed", True) for r in runs
        ),
    }
    for key in ("breakdown", "advance_breakdown"):
        names = list(
            dict.fromkeys(
                s["name"] for r in runs for s in r.get(key, {}).get("stages", ())
            )
        )
        stages = []
        for name in names:
            row = {"name": name}
            for metric in ("wall_us", "thread_cpu_us"):
                values = [
                    next(
                        (
                            s[metric]
                            for s in r.get(key, {}).get("stages", ())
                            if s["name"] == name
                        ),
                        0,
                    )
                    for r in runs
                ]
                mean, sd = mean_sd(values)
                row[metric] = float(mean)
                row[metric + "_sd"] = float(sd)
            stages.append(row)
        summary[key] = {"stages": stages}
    deadlines = [r["deadlines"] for r in runs if r.get("deadlines")]
    if deadlines:
        data = {
            "total_windows": sum(d["windows"] for d in deadlines),
            "total_missed_deadlines": sum(d["missed_deadlines"] for d in deadlines),
            "total_callback_overruns": sum(
                d["callback_budget_overruns"] for d in deadlines
            ),
            "worst_response_ms": max(d["response_ms"]["maximum"] for d in deadlines),
        }
        for metric in (
            "missed_deadlines",
            "miss_rate",
            "callback_budget_overruns",
            "caller_thread_cpu_percent",
        ):
            mean, sd = mean_sd([d.get(metric, 0) for d in deadlines])
            data[metric + "_mean"] = float(mean)
            data[metric + "_sd"] = float(sd)
        summary["deadlines"] = data
    return summary


def interval_seed_series(runs, interval_ms=100):
    """One mean per seed per fixed interval, then mean/SD across seeds.

    Timing means do not determine misses: rates use each original completion.
    All seeds in a group must use the same release schedule and window count.
    """
    per_seed = []
    edges = None
    for run in runs:
        d = run["deadlines"]
        raw = deadline_arrays(
            np.asarray(d["samples"], dtype=np.int64), round(d["period_ms"] * 1e6)
        )
        rates = deadline_rates(
            raw["release_ns"],
            raw["deadline_missed"],
            raw["callback_overrun"],
            interval_ms,
        )
        if edges is not None and not np.array_equal(edges, rates["edges_ms"]):
            raise ValueError("seed release intervals must match")
        edges = rates["edges_ms"]
        bins = raw["release_ns"] // round(interval_ms * 1e6)
        row = {"miss_pct": rates["miss_pct"], "overrun_pct": rates["overrun_pct"]}
        for metric in ("callback_ns", "response_ns"):
            sums = np.bincount(
                bins, weights=raw[metric] / 1e6, minlength=len(rates["windows"])
            )
            row[metric.replace("_ns", "_ms")] = np.divide(
                sums,
                rates["windows"],
                out=np.full(len(sums), np.nan),
                where=rates["windows"] > 0,
            )
        per_seed.append(row)
    result = {"edges_ms": edges, "centres_ms": (edges[:-1] + edges[1:]) / 2}
    for metric in per_seed[0]:
        result[metric], result[metric + "_sd"] = mean_sd(
            [row[metric] for row in per_seed]
        )
    return result


def callback_summary(runs):
    """Pool original paced callbacks, not seed percentiles or interval means."""
    durations = []
    periods = set()
    missed = overruns = 0
    for run in runs:
        deadlines = run.get("deadlines")
        if not deadlines or not deadlines.get("samples"):
            raise ValueError("Callback summary requires paced samples for every seed")
        period = deadlines["period_ms"]
        periods.add(period)
        raw = deadline_arrays(
            np.asarray(deadlines["samples"], dtype=np.int64), round(period * 1e6)
        )
        durations.append(raw["callback_ns"] / 1e6)
        missed += int(raw["deadline_missed"].sum())
        overruns += int(raw["callback_overrun"].sum())
    if len(periods) != 1:
        raise ValueError("Callback summary requires the same deadline budget per seed")
    values = np.concatenate(durations)
    return {
        "windows": len(values),
        "seeds": len(runs),
        "period_ms": periods.pop(),
        "mean_ms": float(values.mean()),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "missed_deadlines": missed,
        "callback_overruns": overruns,
    }


def plot_report(report_path, output=None):
    """Pooled paced callback mean/tails and individual deadline failure counts."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import FuncFormatter, MaxNLocator, ScalarFormatter

    report_path = Path(report_path)
    report = json.loads(report_path.read_text())
    cases = [c for c in report["cases"] if c.get("runs", [c])]
    if not cases:
        return None
    # Legacy reports have no raw paced samples; retain their diagnostic redraw.
    if any(
        not r.get("deadlines", {}).get("samples")
        for c in cases
        for r in c.get("runs", [c])
    ):
        return plot_detailed_report(
            report_path, output or report_path.with_name("streaming.png")
        )
    cases.sort(key=lambda c: c["config"]["hidden"])
    summaries = [callback_summary(c.get("runs", [c])) for c in cases]
    periods = {s["period_ms"] for s in summaries}
    if len(periods) != 1:
        raise ValueError("Summary plot requires the same deadline budget across sizes")
    period = periods.pop()
    sizes = [c["config"]["hidden"] for c in cases]
    x = np.arange(len(sizes))
    fig = Figure(figsize=(11.6, 5.4))
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.26, top=0.79, wspace=0.29)
    ax = axes[0]
    for metric, label, colour, marker in (
        ("mean_ms", "Mean", "#2563a6", "o"),
        ("p95_ms", "p95", "#168477", "s"),
        ("p99_ms", "p99", "#b56a12", "^"),
    ):
        ax.plot(
            sizes,
            [s[metric] for s in summaries],
            label=label,
            color=colour,
            marker=marker,
            linewidth=2,
            markersize=6,
        )
    ax.axhline(period, color="#b72e35", linestyle="--", linewidth=1.6)
    budget_label = (
        "realtime budget"
        if all(c["config"].get("window_ms", period) == period for c in cases)
        else "deadline budget"
    )
    ax.annotate(
        f"{period:g} ms {budget_label}",
        xy=(0.02, period),
        xycoords=("axes fraction", "data"),
        xytext=(0, 6),
        textcoords="offset points",
        ha="left",
        color="#a12630",
        fontsize=11,
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes, [str(size) for size in sizes])
    ax.set_yscale("log")
    values = [s[m] for s in summaries for m in ("mean_ms", "p95_ms", "p99_ms")]
    lower = min(min(values), period) * 0.7
    upper = max(max(values), period) * 1.35
    ticks = [
        unit * 10.0**power
        for power in range(
            int(np.floor(np.log10(lower))), int(np.ceil(np.log10(upper))) + 1
        )
        for unit in (1, 2, 5)
        if lower <= unit * 10.0**power <= upper
    ]
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.minorticks_off()
    ax.set_ylim(lower, upper)
    ax.set_ylabel("Callback time (ms, log scale)", fontsize=12)
    ax.set_title("A   Callback duration", loc="left", fontsize=14, pad=14)
    ax.legend(loc="upper left", ncol=3, frameon=False, fontsize=11)

    ax = axes[1]
    maximum = max(
        s[k] for s in summaries for k in ("missed_deadlines", "callback_overruns")
    )
    positive_counts = [
        s[k]
        for s in summaries
        for k in ("missed_deadlines", "callback_overruns")
        if s[k] > 0
    ]
    log_counts = bool(positive_counts) and maximum / min(positive_counts) >= 100
    for offset, metric, label, colour in (
        (-0.19, "missed_deadlines", "Missed deadline", "#b72e35"),
        (0.19, "callback_overruns", f"Callback > {period:g} ms", "#2563a6"),
    ):
        counts = [s[metric] for s in summaries]
        ax.bar(x + offset, counts, width=0.34, color=colour, label=label, zorder=3)
        for position, count in zip(x + offset, counts, strict=True):
            ax.annotate(
                f"{count / 1000:g}k" if count >= 1000 else str(count),
                (position, count),
                xytext=((-3 if offset < 0 else 3) if count >= 1000 else 0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=10 if count >= 1000 else 12,
                color=colour,
                fontweight="bold",
            )
    windows = {s["windows"] for s in summaries}
    count_label = (
        f"Count / {next(iter(windows)):,} callbacks" if len(windows) == 1 else "Count"
    )
    ax.set_ylabel(count_label + ("\n(log above 1)" if log_counts else ""), fontsize=12)
    ax.set_xticks(x, [str(size) for size in sizes])
    if log_counts:
        ax.set_yscale("symlog", linthresh=1, linscale=0.5)
        ax.set_yticks([0, 1, 10, 100, 1000, 10000, 100000])
        ax.yaxis.set_major_formatter(
            FuncFormatter(
                lambda value, _: f"{value / 1000:g}k" if value >= 1000 else f"{value:g}"
            )
        )
        ax.set_ylim(0, maximum * 4)
    else:
        ax.set_ylim(0, max(maximum, 1) * 1.45)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("B   Deadline failures", loc="left", fontsize=14, pad=14)
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    for ax in axes:
        ax.set_xlabel("Hidden neurons", fontsize=12)
        ax.tick_params(labelsize=11)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    cfg = cases[0]["config"]
    fig.suptitle("Streaming performance as network size increases", fontsize=17, y=0.98)
    sample_text = "; ".join(
        f"{size}: {s['windows']:,} callbacks / {s['seeds']} seeds"
        for size, s in zip(sizes, summaries, strict=True)
    )
    if len(windows) == 1 and len({s["seeds"] for s in summaries}) == 1:
        sample_text = f"{summaries[0]['windows']:,} callbacks per size · {summaries[0]['seeds']} seeds pooled"
    fig.text(
        0.5,
        0.89,
        f"{cfg.get('device', 'cpu').upper()} · {sample_text}",
        ha="center",
        fontsize=11,
    )
    fig.text(
        0.08,
        0.09,
        "Mean / p95 / p99 use original paced callback durations, including input and output processing.\n"
        "Deadline miss: finish after scheduled release + budget. Callback overrun: duration > budget. Counts overlap.\n"
        "A low p99 can coexist with rare overruns. Observed host timings; no hard realtime guarantee.",
        fontsize=10,
        linespacing=1.5,
        va="center",
    )
    output = Path(output) if output else report_path.with_name("streaming.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, facecolor="white")
    return output


def plot_detailed_report(report_path, output=None):
    """Means and SD across seeds; one row per network, four measurement panels."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import PercentFormatter

    report_path = Path(report_path)
    report = json.loads(report_path.read_text())
    groups = []
    for case in report["cases"]:
        runs = case.get("runs", [case])
        if runs:
            groups.append(
                {
                    "config": case["config"],
                    "runs": runs,
                    "summary": summarize_seeds(runs),
                }
            )
    if not groups:
        return None
    height = max(
        5,
        1.7
        + 0.32 * max(len(g["summary"]["advance_breakdown"]["stages"]) for g in groups),
    )
    fig = Figure(figsize=(28, height * len(groups) + 2), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(len(groups), 4, squeeze=False)
    limits = {}
    for key in ("breakdown", "advance_breakdown"):
        limits[key] = (
            max(
                (
                    s[m] + s[m + "_sd"]
                    for g in groups
                    for s in g["summary"][key]["stages"]
                    for m in ("wall_us", "thread_cpu_us")
                ),
                default=1,
            )
            * 1.35
        )
    series = {
        i: interval_seed_series(g["runs"])
        for i, g in enumerate(groups)
        if all(r.get("deadlines") for r in g["runs"])
    }
    time_limit = (
        max(
            (
                max(
                    g["runs"][0]["deadlines"]["period_ms"],
                    float(
                        np.nanmax(
                            series[i]["response_ms"] + series[i]["response_ms_sd"]
                        )
                    ),
                )
                for i, g in enumerate(groups)
                if i in series
            ),
            default=1,
        )
        * 1.1
    )
    rate_limit = min(
        100,
        max(
            0.2,
            max(
                (
                    float(np.nanmax(s[m] + s[m + "_sd"]))
                    for s in series.values()
                    for m in ("miss_pct", "overrun_pct")
                ),
                default=0,
            ),
        )
        * 1.15,
    )
    for index, (row, group) in enumerate(zip(axes, groups, strict=True)):
        hidden = group["config"]["hidden"]
        summary = group["summary"]
        n = summary["seed_count"]
        for column, key in enumerate(("breakdown", "advance_breakdown")):
            ax = row[column]
            stages = summary[key]["stages"]
            y = np.arange(len(stages))
            for offset, metric, colour, label in (
                (-0.18, "wall_us", "#b6cee3", "Wall"),
                (0.18, "thread_cpu_us", "#2c67a2", "Thread CPU"),
            ):
                ax.barh(
                    y + offset,
                    [s[metric] for s in stages],
                    xerr=[s[metric + "_sd"] for s in stages] if n > 1 else None,
                    height=0.34,
                    color=colour,
                    label=label,
                    capsize=2,
                    error_kw={"elinewidth": 0.8},
                )
            for i, stage in enumerate(stages):
                end = max(
                    stage[m] + stage[m + "_sd"] for m in ("wall_us", "thread_cpu_us")
                )
                ax.text(
                    end + limits[key] * 0.012,
                    i,
                    f"{stage['wall_us']:.1f} / {stage['thread_cpu_us']:.1f}",
                    va="center",
                    fontsize=8,
                )
            ax.set_yticks(y, [s["name"] for s in stages], fontsize=9)
            ax.invert_yaxis()
            ax.set_xlim(0, limits[key])
            ax.set_xlabel(
                "Mean µs ± SD across seeds; labels = wall / thread CPU"
                if n > 1
                else "Single-run mean µs; SD unavailable"
            )
            ax.set_title(
                f"{hidden} hidden • "
                + ("Callback breakdown" if column == 0 else "Inside stream.advance")
                + f" • n={n} seeds",
                fontsize=11,
            )
            ax.grid(axis="x", alpha=0.2)
            ax.set_axisbelow(True)
            ax.legend(fontsize=8, loc="lower right")
        if index not in series:
            for ax in row[2:]:
                ax.text(
                    0.5,
                    0.5,
                    "No paced deadline data in this saved report",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_axis_off()
            continue
        s = series[index]
        d = summary["deadlines"]
        ax = row[2]
        for metric, colour, label in (
            ("response_ms", "#df8425", "Release → completion"),
            ("callback_ms", "#2c67a2", "Callback duration"),
        ):
            ax.errorbar(
                s["centres_ms"],
                s[metric],
                yerr=s[metric + "_sd"] if n > 1 else None,
                color=colour,
                label=label,
                linewidth=1,
                capsize=2,
                elinewidth=0.8,
            )
        period = group["runs"][0]["deadlines"]["period_ms"]
        ax.axhline(
            period,
            color="#b72e35",
            linestyle="--",
            linewidth=1,
            label=f"{period:g} ms budget",
        )
        ax.set(
            xlabel="Scheduled release time (ms); 100 ms interval means",
            ylabel="Mean elapsed ms ± SD",
            title=f"{hidden} hidden • {n} seeds; {group['runs'][0]['deadlines']['windows']:,} windows/seed\nWorst individual completion {d['worst_response_ms']:.3f} ms; caller CPU {d['caller_thread_cpu_percent_mean']:.1f}%",
        )
        ax.set_ylim(0, time_limit)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        ax = row[3]
        for metric, colour, label, style in (
            ("miss_pct", "#b72e35", "Deadline miss rate", "-"),
            ("overrun_pct", "#2c67a2", "Callback overrun rate", "--"),
        ):
            ax.errorbar(
                s["centres_ms"],
                s[metric],
                yerr=s[metric + "_sd"] if n > 1 else None,
                color=colour,
                label=label,
                linestyle=style,
                linewidth=1,
                capsize=2,
                elinewidth=0.8,
            )
        ax.set(
            xlabel="Scheduled release time (ms); fixed 100 ms intervals",
            ylabel="Mean fraction of windows ± SD",
            title=f"{hidden} hidden • pooled misses {d['total_missed_deadlines']:,}/{d['total_windows']:,}\nMisses/seed {d['missed_deadlines_mean']:.2f} ± {d['missed_deadlines_sd']:.2f} SD",
        )
        ax.set_ylim(0, rate_limit)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        for ax in row[2:]:
            ax.set_xlim(s["edges_ms"][0], s["edges_ms"][-1])
    cfg = groups[0]["config"]
    fig.suptitle(
        f"Streaming benchmark • {cfg['device'].upper()} • input {cfg.get('rate_hz', 25):g} Hz/channel • spin {cfg.get('spin_us', 0):g} µs\nMeans ± SD across seeds; separate paced deadline and unpaced diagnostic passes",
        fontsize=17,
    )
    fig.supxlabel(
        "SD = sample standard deviation (ddof=1) across seed-level measurements; windows are not independent replicates. Timing panels average within each 100 ms interval before combining seeds.\nMisses are counted from each original window before averaging: mean timing can hide isolated deadline failures. Full totals are inclusive; remaining diagnostic phases sum to them.\nLate windows catch up without dropping steps. Thread CPU excludes workers/device compute. Seed variation also includes host scheduling variation; SD is not a hard-deadline guarantee.",
        fontsize=10,
    )
    output = Path(output) if output else report_path.with_name("streaming-details.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, facecolor="white")
    return output


def main():
    wall_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-only", type=Path, help="Redraw charts from a saved report or directory"
    )
    parser.add_argument(
        "--detailed-plot",
        action="store_true",
        help="Draw the four-panel-per-size diagnostics as streaming-details.png",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Explicit distinct seeds; default is five consecutive seeds starting at --seed (7)",
    )
    parser.add_argument(
        "--hidden-sizes", nargs="+", type=int, default=[64, 128, 256, 512, 1024, 2048]
    )
    for name, kind, default in (
        ("inputs", int, 64),
        ("readouts", int, 8),
        ("sparsity", float, 0.9),
        ("rate-hz", float, 25.0),
        ("dt-ms", float, 0.1),
        ("window-ms", float, 1.0),
        ("period-ms", float, 1.0),
        ("spin-us", float, 1000.0),
        ("windows", int, 256),
        ("seed", int, 7),
        ("block-steps", int, 10),
        ("warmup-windows", int, 20),
        ("threads", int, 1),
    ):
        parser.add_argument(f"--{name}", type=kind, default=default)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument(
        "--compile-mode", choices=("default", "eager"), default="default"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory (overwritten each run); defaults to benchmarks/streaming/results",
    )
    args = vars(parser.parse_args())
    plot_only = args.pop("plot_only")
    plot = plot_detailed_report if args.pop("detailed_plot") else plot_report
    output = args.pop("output")
    if plot_only is not None:
        report_path = plot_only / "report.json" if plot_only.is_dir() else plot_only
        print(plot(report_path))
        return
    if output is None:
        output = Path(__file__).resolve().parent / "results"
    sizes = args.pop("hidden_sizes")
    seeds = args.pop("seeds")
    seeds = list(range(args["seed"], args["seed"] + 5)) if seeds is None else seeds
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        parser.error("--seeds must be distinct nonnegative integers")
    configs = [Config(hidden=size, **args) for size in sizes]
    try:
        for cfg in configs:
            cfg.validate()
    except ValueError as error:
        parser.error(str(error))
    output.mkdir(parents=True, exist_ok=True)
    # Remove prior generated products, including legacy sample files.
    # An explicit output directory may also contain unrelated files; preserve them.
    for pattern in (
        "case-*-breakdown.npz",
        "case-*-advance-phases.npz",
        "case-*-deadlines.npz",
        "report.json",
        "streaming.png",
        "streaming-details.png",
        "breakdown.png",
        "advance-internals.png",
    ):
        for previous in output.glob(pattern):
            if previous.is_file():
                previous.unlink()
    report = {
        "schema": "snnlab.streaming-timings/v5",
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "clock": time.get_clock_info("perf_counter").implementation,
        "scope": "Separate warmed paced deadline pass and unpaced phase diagnostics. Deadlines include input generation, inference, host output extraction and device completion; absolute monotonic releases, no dropped steps.",
        "seeds": seeds,
        "statistics": "Equal-weight mean and SD across seeds; SD uses sample standard deviation (ddof=1), without division by sqrt(n). Per-window samples are not independent seed replicates.",
        "execution_order": [],
        "complete": False,
        "cases": [
            {
                "config": {
                    key: value for key, value in asdict(cfg).items() if key != "seed"
                },
                "runs": [],
            }
            for cfg in configs
        ],
    }
    report_path = output / "report.json"
    for seed_index, seed in enumerate(seeds):
        # Rotate size order to reduce a fixed size/time-of-run confound.
        order = list(range(len(configs)))
        rotation = seed_index % len(order)
        order = order[rotation:] + order[:rotation]
        for index in order:
            cfg = replace(configs[index], seed=seed)
            print(
                f"seed={seed} hidden={cfg.hidden}: warmup, validation and timing",
                flush=True,
            )
            case = report["cases"][index]
            case["runs"].append(run(cfg))
            case["summary"] = summarize_seeds(case["runs"])
            report["execution_order"].append({"seed": seed, "hidden": cfg.hidden})
            report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(plot(report_path), flush=True)
    report["complete"] = True
    report["wall_time_seconds"] = time.perf_counter() - wall_started
    report["wall_time_scope"] = (
        "main entry through final plot generation; excludes Python/import startup and final JSON write"
    )
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Benchmark wall time: {report['wall_time_seconds']:.2f} s", flush=True)


if __name__ == "__main__":
    main()
