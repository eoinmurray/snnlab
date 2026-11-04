"""Small native fine-tuning workflow; synthetic data, no external downloads."""

import json
import shutil
from pathlib import Path

import numpy as np
import torch

from snnlab import extensions, lang, sim

OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def prepare_results():
    if OUTPUT_DIR.is_symlink():
        OUTPUT_DIR.unlink()
    elif OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)


def circuit(*, diagonal=True, cells_count=2):
    net = lang.Network("private_inhibition", dt=0.1 * lang.ms)
    drive = net.input(
        "drive", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
    )
    private = net.input(
        "artificial",
        shape=("time", "batch", cells_count),
        signal_type="continuous",
        unit="1",
    )
    cells = net.population(
        "E", size=cells_count, neuron=lang.COBA_LIF(tau_mem=20 * lang.ms)
    )
    net.connect(
        drive,
        cells.excitatory,
        name="sensory",
        synapse=lang.AMPA(tau=2 * lang.ms),
        weight=lang.Constant(0.3),
        initialization_scaling="direct",
        constraint=lang.NonNegative(),
    )
    net.connect(
        private,
        cells.inhibitory,
        name="private",
        synapse=lang.GABA(tau=6 * lang.ms),
        weight=lang.Constant(0.02),
        initialization_scaling="direct",
        delay=0.1 * lang.ms,
        connectivity=lang.OneToOne() if diagonal else None,
        constraint=lang.NonNegative(),
    )
    scores = lang.MeanVoltage(
        source=cells.spikes,
        classes=2,
        name="scores",
        weight=lang.Constant(0.1),
        initialization_scaling="direct",
    )
    net.output("class_scores", scores)
    recipe = lang.TrainSpec(
        objectives=[lang.CrossEntropy(prediction=scores, target="label")],
        parameter_groups=[
            lang.ParameterGroup(
                ["sensory.weight", "scores_projection.weight"],
                name="learned",
                lr=0.0004,
            ),
            lang.ParameterGroup(["private.weight"], name="private", lr=0, frozen=True),
        ],
        optimizer=lang.AdamW(weight_decay=0),
        gradient_clip=1.0,
        presentation_duration=2 * lang.ms,
    )
    return lang.compile(
        net, training=recipe, target=None, refractory_timing="legacy_coba"
    )


def encode(arrays, selected, *, dt_ms, channels, config, seed, device="cpu", context):
    """Fixed sensory spikes, fresh training histories, repeatable evaluation draws."""
    steps, prehistory = 20, 600
    private_channels = config.get("private_channels", channels)
    drive, artificial, conductance, pending = [], [], [], []
    decay = np.exp(-dt_ms / 6)
    for index, sample_id in zip(selected, context.sample_ids, strict=True):
        sensory_rng = context.generator(
            sample_id, "sensory", policy="fixed", device=device
        )
        inhibition_rng = context.generator(sample_id, "inhibition", device=device)
        probabilities = torch.tensor(np.array(arrays["features"][index]), device=device)
        drive.append(
            (
                torch.rand(steps, channels, generator=sensory_rng, device=device)
                < probabilities
            ).float()
        )
        history = (
            torch.rand(
                prehistory + steps,
                private_channels,
                generator=inhibition_rng,
                device=device,
            )
            < 0.15
        ).float()
        g = torch.zeros(private_channels, device=device)
        for events in history[: prehistory - 1]:
            g = g * decay + 0.02 * events
        conductance.append(g)
        pending.append(history[prehistory - 1])
        artificial.append(history[prehistory:])
    state = sim.PresentationState(
        signature=config["signature"],
        conductances={"private": torch.stack(conductance)},
        input_histories={"artificial": torch.stack(pending).unsqueeze(0)},
        provenance={
            "policy": "inhibition_only",
            "prehistory_ms": 60,
            "rng": "snnlab.sample-rng/v1",
        },
    )
    return sim.PreparedBatch(
        {
            "drive": torch.stack(drive, dim=1),
            "artificial": torch.stack(artificial, dim=1),
        },
        state,
    )


def main():
    prepare_results()
    extensions.register_encoder("examples.private_inhibition/v1", encode)
    bundle = circuit()
    bundle.visualise(OUTPUT_DIR / "network.png")
    source = sim.build(
        bundle=bundle, device="cpu", performance=sim.Performance(compile_mode=None)
    )
    signature = sim.runtime_state_signature(source.model.plan)
    path = OUTPUT_DIR / "synthetic.npz"
    np.savez(
        path,
        features=np.asarray([[0.8, 0.1], [0.1, 0.8]] * 4, dtype=np.float32),
        labels=np.asarray([0, 1] * 4),
        ids=np.arange(8),
    )
    encoder = sim.DatasetEncoder(
        "custom",
        seed=41,
        definition="examples.private_inhibition/v1",
        config={"signature": signature},
        rng_policy="sample_v1",
    )
    binding = sim.DatasetSnapshotBinding(
        path,
        "drive",
        "synthetic",
        "train",
        encoder,
        target_id="label",
        sample_ids_key="ids",
        sample_indices=tuple(range(6)),
    )
    validation = sim.DatasetSnapshotBinding(
        path,
        "drive",
        "synthetic",
        "validation",
        encoder,
        target_id="label",
        sample_ids_key="ids",
        sample_indices=(6, 7),
    )
    result = sim.train(
        bundle=bundle,
        weights=sim.WeightInitialization(source.parameters),
        input_bindings=[binding],
        validation=sim.ValidationSpec(
            input_bindings=[validation], encoding_seeds=(101, 102, 103)
        ),
        epochs=2,
        batch_size=4,
        batching=sim.Batching(order="shuffle", microbatch_size=2),
        metrics=sim.TrainingMetrics(train="evaluate"),
        observations=sim.EpochObservations(population_rates=("E",)),
        device="cpu",
        performance=sim.Performance(compile_mode=None, precision="highest"),
        training_timing=True,
        save_final_checkpoint=OUTPUT_DIR / "final.checkpoint",
        save_selected_checkpoint=OUTPUT_DIR / "selected.checkpoint",
    )
    resolved, _ = sim.resolve_dataset_snapshot_binding(bundle.graph, binding)
    resolved.tensors.initial_state.save(OUTPUT_DIR / "representative.initial-state")
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(result.metrics, default=str, indent=2) + "\n"
    )
    print(json.dumps(result.metrics["batch_counts"], indent=2))


if __name__ == "__main__":
    main()
