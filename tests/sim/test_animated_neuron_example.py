"""The browser replay must be backed by the example's actual simulation."""

import json
import math
import runpy
from pathlib import Path

import numpy as np
import pytest

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/animated-neuron"


@pytest.fixture(scope="module")
def recording():
    module = runpy.run_path(str(EXAMPLE / "animated_neuron.py"))
    _, data, result = module["record_neuron"]()
    return data, result


def test_recording_matches_saved_export_and_actual_results(recording):
    data, result = recording
    assert data == json.loads((EXAMPLE / "recording.json").read_text())
    assert data["schema"] == "snnlab.neuron-replay/v1"
    assert len(data["time_ms"]) == len(data["voltage_mv"]) == 1441
    assert data["time_ms"][0] == 0
    assert data["time_ms"][-1] == data["duration_ms"]
    assert data["voltage_mv"][0] == data["rest_mv"]
    np.testing.assert_array_equal(
        data["voltage_mv"][1:], result.diagnostics["voltage"][:, 0]
    )
    # Shift library slot labels to integration-end labels, not input-start labels.
    spike_steps = np.flatnonzero(result.outputs["spikes"][:, 0])
    np.testing.assert_array_equal(
        data["spike_times_ms"], (spike_steps + 1) * data["dt_ms"]
    )
    assert len(spike_steps) == 7


def test_input_motion_and_one_step_current_are_aligned(recording):
    data, result = recording
    arrivals = np.array(data["input_arrivals_ms"])
    departures = np.array(data["input_departures_ms"])
    np.testing.assert_array_equal(arrivals - departures, 6)
    np.testing.assert_array_equal(np.diff(departures), 12)
    expected = np.zeros(len(data["time_ms"]) - 1)
    expected[np.rint(arrivals / data["dt_ms"]).astype(int)] = data["current_na"]
    np.testing.assert_allclose(
        result.diagnostics["current"][:, 0], expected, atol=1e-6, rtol=0
    )


def test_recorded_voltage_and_spikes_match_independent_lif_recurrence(recording):
    data, result = recording
    voltage = float(data["rest_mv"])
    decay = math.exp(-data["dt_ms"] / data["tau_ms"])
    gain = data["tau_ms"] / data["capacitance_nf"] * (1 - decay)
    expected_voltage, expected_spikes = [voltage], []
    for step, current in enumerate(result.diagnostics["current"][:, 0]):
        candidate = data["rest_mv"] + (voltage - data["rest_mv"]) * decay
        candidate += float(current) * gain
        if candidate >= data["threshold_mv"]:
            voltage = float(data["rest_mv"])
            expected_spikes.append((step + 1) * data["dt_ms"])
        else:
            voltage = candidate
        expected_voltage.append(voltage)
    np.testing.assert_allclose(data["voltage_mv"], expected_voltage, atol=2e-4, rtol=0)
    assert data["spike_times_ms"] == expected_spikes
