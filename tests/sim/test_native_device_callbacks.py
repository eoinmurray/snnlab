"""Native custom objectives and regularizers use the actual tensor device."""

import pytest
import torch

from snnlab import extensions as E
from snnlab import lang
from snnlab.sim import DenseArrayBinding
from snnlab.sim.requests.spec import ExecutionSpec
from snnlab.sim.training.trainer import train


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS unavailable"
            ),
        ),
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_native_custom_loss_and_regularizer_use_resolved_tensor_device(
    device, monkeypatch
):
    monkeypatch.setattr(E, "_REGISTRY", {})
    E.register_operation(
        "test.device_affine/v1",
        lambda sources, parameters, config: sources[0].mean(0) * parameters["gain"],
    )
    E.register_objective(
        "test.device_mse/v1",
        lambda prediction, target, config: (prediction - target).square().mean(),
    )
    E.register_regularizer(
        "test.device_energy/v1",
        lambda signals, duration, config: signals[0].square().mean(),
    )
    net = lang.Network("native-devices", dt=1 * lang.ms)
    source = net.input(
        "values", shape=("time", "batch", 1), signal_type="continuous", unit="1"
    )
    gain = net.parameter("gain", shape=(1,), initializer=lang.Constant(0.1))
    prediction = lang.custom_operation(
        "test.device_affine/v1",
        source,
        name="affine",
        shape=("batch", 1),
        unit="1",
        parameters=(gain,),
    )
    net.output("prediction", prediction)
    recipe = lang.TrainSpec(
        objectives=(
            lang.CustomObjective(
                "test.device_mse/v1", prediction=prediction, target="value"
            ),
        ),
        regularizers=(
            lang.CustomRegularizer(
                "test.device_energy/v1", signals=(prediction,), strength=0.01
            ),
        ),
        parameter_groups=(lang.ParameterGroup((gain,), name="all", lr=0.1),),
        optimizer=lang.AdamW(weight_decay=0),
    )
    bundle = lang.compile(net, training=recipe)
    result = train(
        ExecutionSpec(
            kind="train",
            graph=bundle.graph,
            training=bundle.training,
            input_bindings=(DenseArrayBinding("values", torch.ones(3, 4, 1)),),
            targets={"value": torch.ones(4, 1)},
            device=device,
            epochs=2,
            updates=2,
            batch_size=2,
            diagnostics=False,
        )
    )
    assert result.training_checkpoint.completed_updates == 2
    assert result.parameters["gain"].item() > 0.1
