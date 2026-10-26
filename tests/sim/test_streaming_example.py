"""The tutorial's saved windows must match an uninterrupted eager trajectory."""

import runpy
from pathlib import Path

import numpy as np
import torch

from snnlab import lang
from snnlab.sim import GraphExecutor, plan_graph

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/streaming"


def test_saved_stream_matches_uninterrupted_execution():
    example = runpy.run_path(str(EXAMPLE / "example.py"))
    inputs = torch.cat(list(example["incoming_windows"]()))
    graph = lang.compile(example["make_network"]()).graph
    reference = GraphExecutor(
        plan_graph(graph),
        seed=example["SEED"],
        compile_mode=None,
    )
    with torch.no_grad():
        expected = reference({"events": inputs}).numpy(batch=0)
    with np.load(EXAMPLE / "results/streaming.npz") as saved:
        np.testing.assert_array_equal(saved["input_spikes"], inputs[:, 0].numpy())
        np.testing.assert_allclose(saved["time_ms"], expected.time_ms)
        np.testing.assert_array_equal(saved["spikes"], expected.outputs["spikes"])
        np.testing.assert_allclose(
            saved["voltage_mv"],
            expected.outputs["voltage"],
            atol=3e-4,
            rtol=0,
        )
        assert np.count_nonzero(saved["spikes"]) > 0
