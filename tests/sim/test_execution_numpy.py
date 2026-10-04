"""NumPy conversion preserves named axes, timing and original tensor ownership."""

import numpy as np
import pytest
import torch

from snnlab import lang
from snnlab.sim.execution import (
    DenseArrayBinding,
    ExecutionResult,
    ExecutionSpec,
    PoissonInputBinding,
    simulate,
    train,
)
from tests.sim._execution_builders import direct_train_bundle


def _graph():
    net = lang.Network("numpy_result", dt=0.1 * lang.ms)
    inputs = net.input(
        "inputs", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
    )
    counts = lang.readouts.SpikeCount(source=inputs, classes=2, name="count")
    net.output("spikes", inputs)
    net.output("counts", counts)
    net.expose(inputs, name="input_trace")
    net.expose(counts, name="count_diagnostic")
    return lang.compile(net, target="tools/snnsim").graph


def _result(**options):
    inputs = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]] * 4)
    return simulate(
        ExecutionSpec(
            kind="simulate",
            graph=_graph(),
            input_bindings=(DenseArrayBinding("inputs", inputs),),
            device="cpu",
            **options,
        )
    )


def test_numpy_preserves_all_batches_or_selects_the_declared_axis():
    result = _result()
    all_batches = result.numpy()
    selected = result.numpy(batch=1)
    assert all_batches.outputs["spikes"].shape == (4, 2, 2)
    assert selected.outputs["spikes"].shape == (4, 2)
    assert all_batches.outputs["counts"].shape == (2, 2)
    assert selected.outputs["counts"].shape == (2,)
    for name in ("spikes", "counts"):
        axis = 1 if name == "spikes" else 0
        np.testing.assert_array_equal(
            selected.outputs[name], np.take(all_batches.outputs[name], 1, axis=axis)
        )
    np.testing.assert_array_equal(
        selected.diagnostics["input_trace"], selected.outputs["spikes"]
    )
    np.testing.assert_array_equal(
        selected.diagnostics["count_diagnostic"], selected.outputs["counts"]
    )
    np.testing.assert_allclose(selected.time_ms, [0.0, 0.1, 0.2, 0.3])


def test_numpy_copies_arrays_without_detaching_the_original_computation():
    source = torch.tensor([1.0, 2.0], requires_grad=True)
    result = ExecutionResult(executor="graph", outputs={"value": source * 2})
    data = result.numpy()
    data.outputs["value"][:] = 0
    assert data.time_ms is None
    torch.testing.assert_close(result.outputs["value"], torch.tensor([2.0, 4.0]))
    result.outputs["value"].sum().backward()
    torch.testing.assert_close(source.grad, torch.tensor([2.0, 2.0]))


def test_numpy_timing_follows_continuation_and_diagnostics_can_be_disabled():
    first = _result()
    second = _result(runtime_state=first.runtime_state, diagnostics=False)
    data = second.numpy(batch=0)
    assert data.diagnostics == {}
    np.testing.assert_allclose(data.time_ms, [0.4, 0.5, 0.6, 0.7])


def test_numpy_uses_effective_timestep_after_inference_overrides():
    result = simulate(
        ExecutionSpec(
            kind="infer",
            graph=_graph(),
            device="cpu",
            input_bindings=(
                PoissonInputBinding(
                    input_id="inputs", steps_count=4, rates_hz=(0.0,), seed=1
                ),
            ),
            options={"inference_overrides": {"timestep_ms": 0.2, "duration_ms": 1.0}},
        )
    )
    assert result.numpy().outputs["spikes"].shape == (5, 1, 2)
    np.testing.assert_allclose(result.numpy().time_ms, [0.0, 0.2, 0.4, 0.6, 0.8])


def test_training_retains_numpy_axis_metadata_for_final_update():
    bundle = direct_train_bundle()
    result = train(
        ExecutionSpec(
            kind="train",
            graph=bundle.graph,
            training=bundle.training,
            input_bindings=(DenseArrayBinding("events", torch.ones(3, 2, 2)),),
            targets={"label": torch.tensor([0, 1])},
            device="cpu",
        )
    )
    data = result.numpy(batch=1)
    assert data.outputs["class_scores"].shape == (2,)
    np.testing.assert_array_equal(
        data.outputs["class_scores"], result.outputs["class_scores"][1].detach().numpy()
    )
    np.testing.assert_allclose(data.time_ms, [0.0, 0.1, 0.2])


@pytest.mark.parametrize(
    "batch,error",
    [(-1, IndexError), (2, IndexError), (True, TypeError), (0.5, TypeError)],
)
def test_numpy_rejects_invalid_batch_selection(batch, error):
    with pytest.raises(error, match="batch"):
        _result().numpy(batch=batch)


def test_numpy_requires_metadata_for_batch_selection():
    with pytest.raises(ValueError, match="execution metadata"):
        ExecutionResult(executor="graph").numpy(batch=0)
