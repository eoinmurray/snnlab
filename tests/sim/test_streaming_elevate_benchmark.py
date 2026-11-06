"""Registered adaptive state survives prepared streaming chunk boundaries.

Keep this regression self-contained: the former Elevate benchmark is not part
of the source distribution.
"""

import pytest
import torch

from snnlab import extensions, lang
from snnlab.sim import GraphExecutor, Performance, plan_graph, prepare_stream
from tests.sim.test_execution_extensions import custom_network


@pytest.mark.parametrize("mode", [None, "default"])
def test_custom_streaming_state_and_continuation(monkeypatch, mode):
    monkeypatch.setattr(extensions, "_REGISTRY", {})
    original = torch.compile
    monkeypatch.setattr(
        torch,
        "compile",
        lambda fn, **kw: original(fn, backend="eager", fullgraph=True, dynamic=False),
    )
    net, _, cells, _ = custom_network()
    readout = net.population(
        "readout", size=2, neuron=lang.CustomNeuron("test.adaptive/v1", threshold=0.9)
    )
    net.connect(
        cells.spikes,
        readout.excitatory,
        name="readout_drive",
        synapse=lang.CustomSynapse("test.trace/v1"),
        weight=lang.Constant(1),
    )
    net.output("readout_spikes", readout.spikes)
    graph = lang.compile(net).graph
    values = torch.ones(30, 2, 1)
    policy = Performance(compile_mode=mode, precision="highest", cuda_graphs="off")
    with torch.inference_mode():
        expected = GraphExecutor(plan_graph(graph), compile_mode=None)(
            {"inputs": values}, diagnostics=False
        )
        with prepare_stream(
            graph, device="cpu", window_steps=10, batch_size=2, performance=policy
        ) as stream:
            chunks = []
            for start in range(0, 30, 10):
                chunks.append(stream.advance({"inputs": values[start : start + 10]}))
                if start == 0:
                    snapshot = stream.export_state()
            assert stream.completed_steps == 30
            state = stream.export_state()
            for name in expected.outputs:
                actual = torch.cat([chunk.outputs[name] for chunk in chunks])
                torch.testing.assert_close(
                    actual, expected.outputs[name], atol=0, rtol=0
                )
            assert expected.outputs["spikes"].count_nonzero() > 0
            assert expected.outputs["readout_spikes"].count_nonzero() > 0
            assert state.custom_state
            for name, value in expected.runtime_state.custom_state.items():
                torch.testing.assert_close(
                    state.custom_state[name], value, atol=0, rtol=0
                )
            stream.reset(snapshot)
            for start in (10, 20):
                replayed = stream.advance({"inputs": values[start : start + 10]})
                for name in expected.outputs:
                    torch.testing.assert_close(
                        replayed.outputs[name],
                        chunks[start // 10].outputs[name],
                        atol=0,
                        rtol=0,
                    )
            assert stream.completed_steps == 30
