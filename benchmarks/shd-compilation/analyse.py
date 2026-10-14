"""Summarize measured samples and independent CUDA/Python profiles."""

import gzip
import json
import pstats
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent


def summarize(folder):
    summary = {}
    for kind in ("coba", "ping"):
        report = json.loads((folder / f"{kind}.json").read_text())
        rows = []
        for row in report["rows"]:
            item = {
                "batch": row["batch"],
                "mode": row["mode"],
                "cold_update_s": row["cold_update_s"],
                "second_update_s": row["second_update_s"],
                "peak_allocated_gib": row["peak_allocated_bytes"] / 1024**3,
                **{
                    f"median_{phase}_s": statistics.median(row[f"{phase}_s"])
                    for phase in ("forward", "backward", "update")
                },
            }
            rows.append(item)
        profiles = {}
        for mode in ("custom", "eager", "compiled"):
            trace_path = folder / f"{kind}-{mode}-trace.json"
            if trace_path.exists():
                trace = json.loads(trace_path.read_text())
            else:
                with gzip.open(str(trace_path) + ".gz", "rt") as handle:
                    trace = json.load(handle)
            kernels = [
                event
                for event in trace["traceEvents"]
                if event.get("cat") == "kernel" and "dur" in event
            ]
            products = [
                event
                for event in kernels
                if any(
                    token in event.get("name", "").lower()
                    for token in ("gemm", "cutlass", "cublas")
                )
            ]
            launches = [
                event
                for event in trace["traceEvents"]
                if event.get("name")
                in ("cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel")
            ]
            stats = pstats.Stats(str(folder / f"{kind}-{mode}.pstats"))
            python_functions = [
                {
                    "file": key[0],
                    "line": key[1],
                    "function": key[2],
                    "calls": value[1],
                    "self_s": value[2],
                    "cumulative_s": value[3],
                }
                for key, value in stats.stats.items()
                if key[2] == "_forward" or "run_backward" in key[2]
            ]
            profiles[mode] = {
                "python_profile_total_s": stats.total_tt,
                "python_functions": python_functions,
                "steps": next(
                    row["profile_steps"]
                    for row in report["rows"]
                    if row["mode"] == mode and "profile_steps" in row
                ),
                "kernel_count": len(kernels),
                "kernel_duration_sum_s": sum(e["dur"] for e in kernels) / 1e6,
                "matrix_kernel_count_by_name": len(products),
                "matrix_kernel_duration_sum_s_by_name": sum(e["dur"] for e in products)
                / 1e6,
                "launch_api_count": len(launches),
                "launch_api_duration_sum_s": sum(e.get("dur", 0) for e in launches)
                / 1e6,
            }
        summary[kind] = {
            "torch": report["torch"],
            "device": report["device"],
            "steps": report["steps"],
            "validation": report["validation"],
            "rows": rows,
            "profiles": profiles,
        }
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    summarize(HERE / "results")
