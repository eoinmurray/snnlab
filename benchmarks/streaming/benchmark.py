"""Generate callback and stream.advance timing bar charts for random SNNs."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
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
    rate_hz: float = 100.0
    dt_ms: float = 0.1
    window_ms: float = 1.0
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
        for name in ("dt_ms", "window_ms"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
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
def measure_advance_phases(cfg, session, prefix, windows):
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
    for reference, observed in zip(expected, actual):
        for name, value in reference.items():
            torch.testing.assert_close(observed[name], value, atol=0, rtol=0)
    observed_state = session.export_state()
    for group in STATE_GROUPS:
        for name, value in getattr(expected_state, group).items():
            torch.testing.assert_close(
                getattr(observed_state, group)[name], value, atol=0, rtol=0
            )
    prefix = Path(prefix)
    target = prefix.with_name(prefix.name + "-advance-phases.npz")
    names = ("Full advance", *PHASES, "Other / measurement")
    np.savez_compressed(target, phase_ns=data, phases=np.array(names))
    return {
        "windows": windows,
        "execution_path": warm_result.metrics["execution_path"],
        "numerical_gate_passed": True,
        "statistic": "arithmetic mean per prepared advance",
        "scope": "Separate warmed unpaced pass with phase-boundary wall/thread-CPU clocks and device completion fences; no PyTorch profiler or cProfile active.",
        "limits": "Phases are nonoverlapping. Full advance is inclusive. Other includes timer bookkeeping and API entry/exit. Compiled block includes dispatch, input views, invocation and completion wait, not just numerical kernels. Thread CPU excludes workers/device compute. Instrumentation changes timing.",
        "samples_file": target.name,
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


def measure_breakdown(cfg, model, prefix):
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

    target = prefix.with_name(prefix.name + "-breakdown.npz")
    np.savez_compressed(target, stage_ns=rows, stages=np.array(BREAKDOWN_STAGES))
    return {
        "windows": cfg.windows,
        "samples_file": target.name,
        "stages": [
            {
                "name": name,
                "wall_us": float(rows[:, i, 0].mean() / 1000),
                "thread_cpu_us": float(rows[:, i, 1].mean() / 1000),
            }
            for i, name in enumerate(BREAKDOWN_STAGES)
        ],
    }


@torch.inference_mode()
def run(cfg, prefix):
    cfg.validate()
    prefix = Path(prefix)
    torch.set_num_threads(cfg.threads)
    torch.set_float32_matmul_precision("highest")
    plan = plan_graph(circuit(cfg))
    with stream(cfg, plan) as model:
        validation = warm_and_validate(cfg, model, plan)
        breakdown = measure_breakdown(cfg, model, prefix)
        internal = measure_advance_phases(cfg, model, prefix, cfg.windows)
        return {
            "config": asdict(cfg),
            "validation": validation,
            "compiler_wrapper": model.info["compiler_wrapper"],
            "breakdown": breakdown,
            "advance_breakdown": internal,
        }


def plot_breakdown(report_path, output=None, *, internal=False):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    report_path = Path(report_path)
    report = json.loads(report_path.read_text())
    if internal:
        cases = []
        for case in report["cases"]:
            data = case.get("advance_breakdown")
            if data:
                cases.append(
                    {
                        **case,
                        "breakdown": {
                            **data,
                        },
                    }
                )
    else:
        cases = [
            case
            for case in report["cases"]
            if (case.get("breakdown") or {}).get("stages")
        ]
    if not cases:
        return None
    columns = min(2, len(cases))
    rows = (len(cases) + columns - 1) // columns
    panel_height = 2 + max(len(case["breakdown"]["stages"]) for case in cases) * 0.4
    fig = Figure(figsize=(8 * columns, panel_height * rows + 1), layout="constrained")
    FigureCanvasAgg(fig)
    axes = list(fig.subplots(rows, columns, squeeze=False).flat)
    limit = (
        max(
            max(stage["wall_us"], stage["thread_cpu_us"])
            for case in cases
            for stage in case["breakdown"]["stages"]
        )
        * 1.35
    )
    for ax, case in zip(axes, cases):
        data = case["breakdown"]
        stages = data["stages"]
        y = np.arange(len(stages))
        wall = [s["wall_us"] for s in stages]
        cpu = [s["thread_cpu_us"] for s in stages]
        ax.barh(y - 0.18, wall, height=0.34, color="#b6cee3", label="Elapsed wall time")
        ax.barh(
            y + 0.18, cpu, height=0.34, color="#2c67a2", label="Calling-thread CPU time"
        )
        for i, (w, c) in enumerate(zip(wall, cpu)):
            ax.text(
                max(w, c) + limit * 0.012,
                i,
                f"{w:.1f} / {c:.1f}",
                va="center",
                fontsize=9,
            )
        ax.set_yticks(y, [s["name"] for s in stages])
        ax.invert_yaxis()
        ax.set_xlim(0, limit)
        ax.set_xlabel(
            "Mean time per advance (µs); labels = wall / thread CPU"
            if internal
            else "Mean time per callback (µs); labels = wall / thread CPU"
        )
        title = f"{case['config']['hidden']} hidden neurons — diagnostic windows n={data['windows']:,}"
        if data.get("aborted", False):
            title += " (PARTIAL)"
        ax.set_title(title, fontsize=12)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        ax.legend(fontsize=9, loc="lower right")
    for ax in axes[len(cases) :]:
        ax.set_visible(False)
    fig.suptitle(
        "Inside stream.advance\nExclusive phases; separate warmed diagnostic pass"
        if internal
        else "Where window processing time is spent\nWarmed unpaced diagnostic pass; arithmetic means",
        fontsize=16,
    )
    fig.supxlabel(
        "Phases + other sum to full advance. Compiled block includes dispatch and completion waits, not just numerical kernels.\nInstrumentation perturbs timing; thread CPU excludes workers/GPU compute. Input generation and host event decoding excluded."
        if internal
        else "Executor includes SNNLab, Python, PyTorch and completion waits; it does not isolate kernels.\nThread CPU excludes worker threads and GPU compute. Substages + other sum to full callback; full callback is not an extra stage.",
        fontsize=10,
    )
    output = (
        Path(output)
        if output
        else report_path.with_name(
            "advance-internals.png" if internal else "breakdown.png"
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, facecolor="white")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-only", type=Path, help="Redraw charts from a saved report or directory"
    )
    parser.add_argument(
        "--hidden-sizes", nargs="+", type=int, default=[64, 128, 256, 512]
    )
    for name, kind, default in (
        ("inputs", int, 64),
        ("readouts", int, 8),
        ("sparsity", float, 0.9),
        ("rate-hz", float, 100.0),
        ("dt-ms", float, 0.1),
        ("window-ms", float, 1.0),
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
        help="New directory for charts and measurements",
    )
    args = vars(parser.parse_args())
    plot_only = args.pop("plot_only")
    output = args.pop("output")
    if plot_only is not None:
        report_path = plot_only / "report.json" if plot_only.is_dir() else plot_only
        for internal in (False, True):
            print(plot_breakdown(report_path, internal=internal))
        return
    if output is None:
        parser.error("--output is required unless using --plot-only")
    sizes = args.pop("hidden_sizes")
    configs = [Config(hidden=size, **args) for size in sizes]
    try:
        for cfg in configs:
            cfg.validate()
    except ValueError as error:
        parser.error(str(error))
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "snnlab.streaming-timings/v1",
        "torch_version": torch.__version__,
        "scope": "Separate warmed, unpaced diagnostic passes; instrumented wall and calling-thread CPU times. No deadline measurement.",
        "cases": [],
    }
    for index, cfg in enumerate(configs):
        print(f"hidden={cfg.hidden}: warmup, validation and timing", flush=True)
        report["cases"].append(
            run(cfg, output / f"case-{index:03d}-hidden-{cfg.hidden}")
        )
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for internal in (False, True):
        print(plot_breakdown(output / "report.json", internal=internal), flush=True)


if __name__ == "__main__":
    main()
