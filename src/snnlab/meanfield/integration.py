"""Adaptive method of steps using retained solve_ivp dense interpolants."""

from bisect import bisect_left
from decimal import ROUND_FLOOR, Decimal
from heapq import heappop, heappush

import numpy as np
from scipy.integrate import solve_ivp


def boundaries(duration, delays, points):
    """Propagate known derivative discontinuities through all delay sums.

    Inputs are quantized once to integer ticks of 1e-12 ms. Delay sums and
    deduplication then use exact integer arithmetic, without accumulating float
    rounding error. A complexity guard still rejects excessive distinct times.
    """
    ticks_per_ms = 10**12

    def ticks(time):
        return int(round(Decimal(str(time)) * ticks_per_ms))

    delay_ticks = sorted({ticks(delay) for delay in delays})
    if any(delay <= 0 for delay in delay_ticks):
        raise ValueError("delay is below the 1e-12 ms boundary resolution")
    # Keep the actual endpoint, including a final fraction of a tick, without
    # admitting quantized discontinuities beyond it.
    end_tick = int(
        (Decimal(str(duration)) * ticks_per_ms).to_integral_value(rounding=ROUND_FLOOR)
    )
    pending, seen = [], set()

    def add(time):
        if time <= end_tick and time not in seen:
            if len(seen) >= 100_000:
                raise ValueError(
                    "too many delay discontinuities; shorten duration or simplify delays"
                )
            seen.add(time)
            heappush(pending, time)

    add(0)
    for point in points:
        point_tick = ticks(point)
        if point_tick >= -max(delay_ticks, default=0):
            add(point_tick)
    while pending:
        time = heappop(pending)
        for delay in delay_ticks:
            add(time + delay)
    return sorted(
        {
            0.0,
            duration,
            *(
                t / ticks_per_ms
                for t in seen
                if 0 < t <= end_tick and t / ticks_per_ms < duration
            ),
        }
    )


def solve_segments(rhs, initial, *, duration, times, delays, points, history, options):
    stops = boundaries(duration, delays, points)
    ends, solutions = [], []
    minimum = min(delays, default=np.inf)
    state = initial
    next_step = None

    def past(time):
        if time <= 0:
            return initial if time == 0 else history(time)
        if not ends or time > ends[-1] + 1e-10:
            raise RuntimeError("delayed evaluation requested uncomputed future state")
        index = min(bisect_left(ends, time), len(solutions) - 1)
        return solutions[index](min(time, ends[-1]))

    for left, stop in zip(stops[:-1], stops[1:]):
        while left < stop:
            right = min(stop, left + minimum)
            if abs(right - stop) <= 8 * abs(np.spacing(stop)):
                right = stop
            if right <= left:
                raise ValueError(
                    "delay is too small for the requested integration horizon"
                )
            interior_left = np.nextafter(left, right)
            interior_right = np.nextafter(right, left)

            def wrapped(time, vector):
                # Use each segment's interior limit at both endpoints: the
                # following segment evaluates the new control value. State is
                # continuous, so this affects only the derivative at a switch.
                evaluation_time = min(max(time, interior_left), interior_right)
                return rhs(evaluation_time, vector, past)

            segment_options = options
            if next_step is not None:
                segment_options = {
                    **options,
                    "first_step": min(next_step, right - left),
                }
            result = solve_ivp(
                wrapped, (left, right), state, dense_output=True, **segment_options
            )
            if not result.success or not np.isfinite(result.y).all():
                raise RuntimeError(f"integration failed: {result.message}")
            ends.append(right)
            # A one-step segment needs no OdeSolution search or dispatch.
            solutions.append(
                result.sol.interpolants[0]
                if len(result.sol.interpolants) == 1
                else result.sol
            )
            if options["method"] in ("RK23", "RK45", "DOP853"):
                # Grow from an accepted step instead of reselecting from zero.
                # The solver still applies its usual local error control.
                next_step = min(options["max_step"], 2 * max(np.diff(result.t)))
            state = result.y[:, -1]
            left = right
    output = np.column_stack([initial if time == 0 else past(time) for time in times])
    return output, len(ends)
