"""Compare adaptive delayed integration against a Git baseline and an ODE.

Run: uv run python benchmarks/meanfield-delays/benchmark.py --baseline-ref COMMIT
Only integration is timed; model construction, symbolic compilation and comparison
are excluded. Delays remain explicit in both delayed runs. The ODE omits delays
and therefore represents different dynamics.
"""

import argparse
import json
import platform
import runpy
import statistics
import subprocess
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import scipy

from snnlab import meanfield as mf
from snnlab.meanfield import integration

ROOT = Path(__file__).resolve().parents[2]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def baseline_module(commit, name):
    source = git("show", f"{commit}:src/snnlab/meanfield/{name}.py")
    qualified = f"snnlab.meanfield._benchmark_baseline_{name}"
    module = types.ModuleType(qualified)
    module.__package__ = "snnlab.meanfield"
    sys.modules[qualified] = module
    exec(compile(source, f"{commit}/{name}.py", "exec"), module.__dict__)
    return module


def run(*, baseline_ref, repeats, duration, output):
    if repeats < 1 or duration <= 0:
        raise ValueError("repeats and duration must be positive")
    commit = git("rev-parse", "--verify", baseline_ref)
    baseline_analysis = baseline_module(commit, "analysis")
    baseline_integration = baseline_module(commit, "integration")
    build = runpy.run_path(str(ROOT / "examples/meanfield-delays/example.py"))[
        "build_model"
    ]
    delayed, _ = build()
    omitted, _ = build(delays="omit")
    # Exclude lazy symbolic compilation equally for all implementations.
    for model in (delayed, omitted):
        _ = model.numerical_rhs
    initial = dict.fromkeys(delayed.states, 0.0)
    initial.update({"E_A.rate": 5, "I_A.rate": 10, "E_B.rate": 7, "I_B.rate": 10})
    settings = dict(
        duration=duration,
        initial=initial,
        sample_interval=0.5,
        method="DOP853",
        max_step=0.1,
        rtol=1e-8,
        atol=1e-10,
    )
    current_solver = integration.solve_segments
    cases = {
        "baseline_delayed": (
            baseline_analysis.integrate,
            baseline_integration.solve_segments,
            delayed,
        ),
        "optimised_delayed": (mf.integrate, current_solver, delayed),
        "baseline_ode": (
            baseline_analysis.integrate,
            baseline_integration.solve_segments,
            omitted,
        ),
        "optimised_ode": (mf.integrate, current_solver, omitted),
    }
    timings = {name: [] for name in cases}
    trajectories = {}
    for repeat in range(repeats):
        order = list(cases)
        if repeat % 2:
            order.reverse()
        for name in order:
            integrate, solver, model = cases[name]
            history = {"history": mf.ConstantHistory(initial)} if model.delayed else {}
            with patch.object(integration, "solve_segments", solver):
                start = perf_counter()
                trajectory = integrate(model, **settings, **history)
                seconds = perf_counter() - start
            timings[name].append(seconds)
            trajectories[name] = trajectory
            print(f"{repeat + 1}/{repeats} {name}: {seconds:.4f} s", flush=True)

    comparisons = {}
    for kind in ("delayed", "ode"):
        before = trajectories[f"baseline_{kind}"]
        after = trajectories[f"optimised_{kind}"]
        np.testing.assert_array_equal(before.time_ms, after.time_ms)
        comparisons[kind] = {}
        for name in before.state:
            original, updated = before.state[name], after.state[name]
            np.testing.assert_allclose(updated, original, rtol=1e-9, atol=1e-10)
            comparisons[kind][name] = {
                "unit": delayed.units[name],
                "max_absolute_difference": float(np.max(np.abs(updated - original))),
            }

    medians = {name: statistics.median(values) for name, values in timings.items()}
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_commit": commit,
        "working_tree_commit": git("rev-parse", "HEAD"),
        "working_tree_changed_files": git("status", "--short"),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "settings": settings,
        "delays_ms": dict(delayed.projection_delays),
        "state_count": len(delayed.states),
        "repeats": repeats,
        "seconds": timings,
        "median_seconds": medians,
        "ratios": {
            "baseline_delay_penalty": medians["baseline_delayed"]
            / medians["baseline_ode"],
            "optimised_delay_penalty": medians["optimised_delayed"]
            / medians["optimised_ode"],
            "delayed_speedup": medians["baseline_delayed"]
            / medians["optimised_delayed"],
            "delayed_time_reduction_fraction": 1
            - medians["optimised_delayed"] / medians["baseline_delayed"],
        },
        "segments": {name: t.metadata["segments"] for name, t in trajectories.items()},
        "trajectory_comparison": {
            "rtol": 1e-9,
            "atol": 1e-10,
            "per_state": comparisons,
        },
        "scope": "Local host integration only. Omitted delays change the equations. No general speed guarantee.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["ratios"], indent=2))
    print(f"Report: {output.resolve()}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--duration", type=float, default=600)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "results/report.json"
    )
    run(**vars(parser.parse_args()))
