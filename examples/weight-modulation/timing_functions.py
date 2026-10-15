"""Portable custom timing function; keep this module importable with the bundle."""

import math


def gated_sine(t_ms, config):
    if t_ms < config["start_ms"]:
        return 0.0
    elapsed_s = (t_ms - config["start_ms"]) / 1000
    return 1 + config["depth"] * math.sin(
        2 * math.pi * config["frequency_hz"] * elapsed_s
    )
