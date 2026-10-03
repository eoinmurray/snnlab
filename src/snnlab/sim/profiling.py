"""Lightweight timing accumulator for the sim / render / encode phases.

Pulled out of plot.py — it's a stopwatch utility, not plotting code. The
shared ``prof`` instance is used by the scan runners to print a
per-phase breakdown after a run.
"""

from __future__ import annotations

import time as _time
from contextlib import contextmanager


class _Profiler:
    """Lightweight accumulator for sim / render / encode timings."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sim = 0.0
        self.render = 0.0
        self.encode = 0.0

    @contextmanager
    def track_sim(self):
        t0 = _time.monotonic()
        yield
        self.sim += _time.monotonic() - t0

    @contextmanager
    def track_render(self):
        t0 = _time.monotonic()
        yield
        self.render += _time.monotonic() - t0

    @contextmanager
    def track_encode(self):
        t0 = _time.monotonic()
        yield
        self.encode += _time.monotonic() - t0

    def report(self, n_frames=None):
        total = self.sim + self.render + self.encode
        if total == 0:
            return
        n = n_frames or 1

        def _bar(val, label, width=20):
            frac = val / total if total > 0 else 0
            filled = int(frac * width)
            bar = "\u2588" * filled + "\u2591" * (width - filled)
            avg = val / n
            return (
                f"  {label:>8s} {bar} {val:>6.1f}s  ({frac:>4.0%})"
                f"  avg {avg * 1000:>6.1f}ms/frame"
            )

        print(
            "  \u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510"
        )
        print(_bar(self.sim, "sim"))
        print(_bar(self.render, "render"))
        print(_bar(self.encode, "encode"))
        fps_str = ""
        if n_frames and total > 0:
            fps_str = f"  {n_frames / total:.1f} fps"
        print(
            f"  \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 total {total:>6.1f}s{fps_str} \u2500\u2500\u2518"
        )


prof = _Profiler()
