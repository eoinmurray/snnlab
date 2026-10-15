"""Importable timing functions used by portable gain tests."""


def configured_step(t_ms, config):
    return config.get("after", 0.0) if t_ms >= config.get("at_ms", 2.0) else 1.0


def mutating_config(t_ms, config):
    config["nested"]["calls"] += 1
    return config["nested"]["calls"]


def invalid_gain(t_ms, config):
    return config["value"]


def broken(t_ms, config):
    raise RuntimeError("timing callback failed")
