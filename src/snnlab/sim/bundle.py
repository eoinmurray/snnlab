"""Translate the first supported snnlang graph subset into legacy CLI settings.

This module deliberately does not import snnlang.  Archived data-only bundles
remain executable without the authoring compiler installed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from snnlab.sim.timing import refractory_steps


class BundleCompatibilityError(ValueError):
    """The bundle is valid data, but outside this backend adapter's subset."""


@dataclass(frozen=True)
class LegacySettings:
    dt: float
    refractory_e_ms: float
    refractory_i_ms: float
    hidden_size: int
    input_size: int
    output_size: int
    w_in: tuple[float, float]
    w_in_i: tuple[float, float] | None
    w_ee: tuple[float, float]
    w_ei: tuple[float, float]
    w_ie: tuple[float, float]
    w_ii: tuple[float, float]
    recurrent_initial_zero_fraction: float
    exact_k_initialization: bool
    tau_gaba: float
    readout_mode: str


@dataclass(frozen=True)
class TrainingSettings:
    lr: float
    weight_decay: float
    epochs: int
    surrogate_slope: float
    voltage_grad_dampen: float
    presentation_duration_ms: float | None


@dataclass(frozen=True)
class BackendCapability:
    """Versioned capability requirement attached to one graph element."""

    schema: str
    element: str
    feature: str


def required_capabilities_v1(graph: dict[str, Any]) -> tuple[BackendCapability, ...]:
    """Describe bundle requirements without importing the authoring package."""
    rows: list[BackendCapability] = []
    for pop in graph.get("populations", []):
        rows.append(
            BackendCapability(
                "tools/snnsim.capability/v1",
                pop["id"],
                f"neuron:{pop['neuron']['kind']}",
            )
        )
    for projection in graph.get("projections", []):
        rows.extend(
            (
                BackendCapability(
                    "tools/snnsim.capability/v1",
                    projection["id"],
                    f"synapse:{projection['synapse']['kind']}",
                ),
                BackendCapability(
                    "tools/snnsim.capability/v1",
                    projection["id"],
                    f"connection:{projection['connection']}",
                ),
                BackendCapability(
                    "tools/snnsim.capability/v1",
                    projection["id"],
                    "delay:integer_steps",
                ),
            )
        )
    for operation in graph.get("operations", []):
        rows.append(
            BackendCapability(
                "tools/snnsim.capability/v1",
                operation["id"],
                f"operation:{operation['kind']}",
            )
        )
    for observable in graph.get("observables", []):
        rows.append(
            BackendCapability(
                "tools/snnsim.capability/v1",
                observable["id"],
                f"recording:{observable['signal'].partition('.')[2]}",
            )
        )
    return tuple(rows)


def _canonical_json(data: Any) -> bytes:
    return (
        json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode()


def _digest(data: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(data)).hexdigest()


def load_graph_bundle(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and authenticate manifest.json + graph.json from a bundle directory."""
    root = Path(path)
    if root.is_file():
        if root.name != "manifest.json":
            raise BundleCompatibilityError(
                "--bundle must name a bundle directory or its manifest.json"
            )
        root = root.parent
    manifest_path = root / "manifest.json"
    graph_path = root / "graph.json"
    if not manifest_path.is_file() or not graph_path.is_file():
        raise BundleCompatibilityError(
            f"{root} is not a bundle: manifest.json and graph.json are required"
        )
    manifest = json.loads(manifest_path.read_text())
    graph = json.loads(graph_path.read_text())
    if manifest.get("schema") != "snnlang.bundle/v1":
        raise BundleCompatibilityError(
            f"unsupported bundle schema: {manifest.get('schema')!r}"
        )
    if graph.get("schema") != "snnlang.graph/v1":
        raise BundleCompatibilityError(
            f"unsupported graph schema: {graph.get('schema')!r}"
        )
    actual = _digest(graph)
    if actual != manifest.get("graph_digest"):
        raise BundleCompatibilityError("graph.json digest does not match manifest.json")
    return manifest, graph


def load_training_recipe(
    path: str | Path, manifest: dict[str, Any], graph: dict[str, Any]
) -> dict[str, Any]:
    """Load and authenticate the optional training recipe required by train."""
    root = Path(path)
    if root.is_file():
        root = root.parent
    training_path = root / "training.json"
    if not training_path.is_file():
        raise BundleCompatibilityError("bundle-driven training requires training.json")
    training = json.loads(training_path.read_text())
    if training.get("schema") != "snnlang.training/v1":
        raise BundleCompatibilityError(
            f"unsupported training schema: {training.get('schema')!r}"
        )
    if training.get("graph_digest") != manifest.get("graph_digest"):
        raise BundleCompatibilityError(
            "training.json graph digest does not match graph.json"
        )
    declared = {row.get("path"): row.get("digest") for row in manifest.get("files", [])}
    if declared.get("training.json") != _digest(training):
        raise BundleCompatibilityError(
            "training.json digest does not match manifest.json"
        )
    if training["graph_digest"] != _digest(graph):
        raise BundleCompatibilityError(
            "training.json graph digest does not authenticate this graph"
        )
    return training


def load_simulation_recipe(
    path: str | Path, manifest: dict[str, Any], graph: dict[str, Any]
) -> dict[str, Any] | None:
    """Load and authenticate an optional bundle-owned simulation recipe."""
    root = Path(path)
    if root.is_file():
        root = root.parent
    recipe_path = root / "simulation.json"
    declared = {row.get("path"): row.get("digest") for row in manifest.get("files", [])}
    if not recipe_path.is_file():
        if "simulation.json" in declared:
            raise BundleCompatibilityError("declared simulation.json is missing")
        return None
    recipe = json.loads(recipe_path.read_text())
    if recipe.get("schema") != "snnlang.simulation/v1":
        raise BundleCompatibilityError(
            f"unsupported simulation schema: {recipe.get('schema')!r}"
        )
    if recipe.get("graph_digest") != manifest.get("graph_digest") or recipe.get(
        "graph_digest"
    ) != _digest(graph):
        raise BundleCompatibilityError(
            "simulation.json graph digest does not authenticate this graph"
        )
    if declared.get("simulation.json") != _digest(recipe):
        raise BundleCompatibilityError(
            "simulation.json digest does not match manifest.json"
        )
    return recipe


def _normal_details(
    projection: dict[str, Any], parameters: dict[str, Any]
) -> tuple[tuple[float, float], float, str]:
    ids = projection.get("parameters", [])
    if len(ids) != 1 or ids[0] not in parameters:
        raise BundleCompatibilityError(
            f"projection {projection.get('id')} must own exactly one weight parameter"
        )
    scaling = parameters[ids[0]].get("initialization_scaling", "fan_in_normalized")
    if scaling != "fan_in_normalized":
        raise BundleCompatibilityError(
            f"projection {projection.get('id')} initialization_scaling={scaling!r} "
            "requires the graph executor; legacy bundle translation supports fan_in_normalized"
        )
    initializer = parameters[ids[0]].get("initializer", {})
    if initializer.get("kind") not in {"normal", "lower_clamped_normal"}:
        raise BundleCompatibilityError(
            f"projection {projection.get('id')} requires a lower-clamped normal initializer"
        )
    fraction = float(initializer.get("initial_zero_fraction", 0.0))
    zeroing = initializer.get("zeroing", "bernoulli")
    if not 0.0 <= fraction < 1.0:
        raise BundleCompatibilityError(
            f"projection {projection.get('id')} has invalid initial_zero_fraction"
        )
    if zeroing not in {"bernoulli", "exact_k"}:
        raise BundleCompatibilityError(
            f"projection {projection.get('id')} has unsupported zeroing {zeroing!r}"
        )
    return (float(initializer["mean"]), float(initializer["std"])), fraction, zeroing


def _normal(
    projection: dict[str, Any], parameters: dict[str, Any]
) -> tuple[float, float]:
    return _normal_details(projection, parameters)[0]


def translate_cobanet_v1(graph: dict[str, Any]) -> LegacySettings:
    """Recognise the exact one-layer PING + mean-voltage COBANet subset."""
    populations = {row["id"]: row for row in graph.get("populations", [])}
    parameters = {row["id"]: row for row in graph.get("parameters", [])}
    projections = graph.get("projections", [])
    operations = {row["id"]: row for row in graph.get("operations", [])}
    inputs = graph.get("inputs", [])
    outputs = graph.get("outputs", [])

    spiking = [row for row in populations.values() if row.get("spiking")]
    analogue = [row for row in populations.values() if not row.get("spiking")]
    if len(spiking) != 2 or len(analogue) != 1:
        raise BundleCompatibilityError(
            "COBANet v1 requires exactly two spiking E/I populations and one "
            "non-spiking readout population"
        )
    if len(inputs) not in {1, 2} or len(outputs) != 1:
        raise BundleCompatibilityError(
            "COBANet v1 requires one or two inputs and one named output"
        )

    inhibitory = [
        row
        for row in projections
        if row.get("polarity") == "inhibitory"
        and row.get("source", "").partition(".")[0]
        != row.get("target", "").partition(".")[0]
    ]
    recurrent_exc = [
        row
        for row in projections
        if row.get("connection") == "recurrent"
        and row.get("polarity") == "excitatory"
        and row.get("source", "").partition(".")[0]
        != row.get("target", "").partition(".")[0]
    ]
    if len(inhibitory) != 1 or len(recurrent_exc) != 1:
        raise BundleCompatibilityError(
            "COBANet v1 requires one recurrent E→I and one recurrent I→E projection"
        )
    e_id = inhibitory[0]["target"].partition(".")[0]
    i_id = inhibitory[0]["source"].partition(".")[0]
    if (
        recurrent_exc[0]["source"].partition(".")[0] != e_id
        or recurrent_exc[0]["target"].partition(".")[0] != i_id
    ):
        raise BundleCompatibilityError("recurrent projections do not form one E↔I loop")
    e_pop, i_pop = populations[e_id], populations[i_id]
    if i_pop["size"] != e_pop["size"] // 4:
        raise BundleCompatibilityError(
            "current COBANet backend requires inhibitory size = excitatory size / 4"
        )
    e_neuron, i_neuron = e_pop.get("neuron", {}), i_pop.get("neuron", {})
    if (
        e_neuron.get("kind") != "coba_lif"
        or e_neuron.get("tau_mem") != {"value": 20.0, "unit": "ms"}
        or i_neuron.get("kind") != "coba_lif"
        or i_neuron.get("tau_mem")
        not in (
            {"value": 5.0, "unit": "ms"},
            # Compatibility with pre-audit bundles. Their legacy route has
            # always executed the historical 5 ms I cell.
            {"value": 10.0, "unit": "ms"},
        )
    ):
        raise BundleCompatibilityError(
            "current COBANet backend requires 20 ms E and 5 ms I COBA-LIF neurons"
        )

    input_ids = {row["id"] for row in inputs}
    input_projection = [
        row
        for row in projections
        if row["source"].partition(".")[0] in input_ids
        and row["target"] == f"{e_id}.excitatory"
    ]
    input_projection_i = [
        row
        for row in projections
        if row["source"].partition(".")[0] in input_ids
        and row["target"] == f"{i_id}.excitatory"
    ]
    readout_id = analogue[0]["id"]
    readout_projection = [
        row
        for row in projections
        if row["source"] == f"{e_id}.spikes"
        and row["target"] == f"{readout_id}.excitatory"
    ]
    if (
        len(input_projection) != 1
        or len(input_projection_i) > 1
        or len(readout_projection) != 1
    ):
        raise BundleCompatibilityError(
            "COBANet v1 requires direct input→E and E→readout projections"
        )
    supported_ampa = (
        {"kind": "ampa", "tau": {"value": 2.0, "unit": "ms"}},
        # Pre-audit bundles declared 5 ms while the legacy kernel used 2 ms.
        {"kind": "ampa", "tau": {"value": 5.0, "unit": "ms"}},
    )
    if (
        input_projection[0].get("synapse") not in supported_ampa
        or (
            input_projection_i
            and input_projection_i[0].get("synapse") not in supported_ampa
        )
        or recurrent_exc[0].get("synapse") not in supported_ampa
    ):
        raise BundleCompatibilityError(
            "current COBANet backend requires 2 ms AMPA input and E→I synapses"
        )
    supported_projection_ids = {
        input_projection[0]["id"],
        readout_projection[0]["id"],
        recurrent_exc[0]["id"],
        inhibitory[0]["id"],
    }
    if input_projection_i:
        supported_projection_ids.add(input_projection_i[0]["id"])
    same_population = [
        row
        for row in projections
        if row["source"].partition(".")[0] == row["target"].partition(".")[0]
    ]
    same_by_population = {
        row["source"].partition(".")[0]: row for row in same_population
    }
    if len(same_by_population) != len(same_population):
        raise BundleCompatibilityError(
            "COBANet v1 allows at most one recurrent projection per population"
        )
    w_ee_row = same_by_population.get(e_id)
    w_ii_row = same_by_population.get(i_id)
    for row in same_population:
        supported_projection_ids.add(row["id"])
    extras = {row["id"] for row in projections} - supported_projection_ids
    if extras:
        raise BundleCompatibilityError(
            f"unsupported projections for COBANet v1: {', '.join(sorted(extras))}"
        )

    output_signal = outputs[0]["signal"]
    output_op = operations.get(output_signal.partition(".")[0])
    if (
        not output_op
        or output_op.get("kind") != "reduce_mean"
        or output_op.get("sources") != [f"{readout_id}.voltage"]
    ):
        raise BundleCompatibilityError(
            "COBANet v1 currently supports only MeanVoltage named outputs"
        )
    if set(operations) != {output_op["id"]}:
        raise BundleCompatibilityError(
            "COBANet v1 does not support additional graph operations"
        )

    dt = graph.get("timebase", {}).get("dt", {})
    if dt.get("unit") != "ms" or float(dt.get("value", 0)) <= 0:
        raise BundleCompatibilityError(
            "graph timebase must declare a positive dt in ms"
        )
    input_shapes = [row.get("shape", []) for row in inputs]
    if (
        any(
            len(shape) != 3
            or shape[:2] != ["time", "batch"]
            or not isinstance(shape[-1], int)
            for shape in input_shapes
        )
        or len({shape[-1] for shape in input_shapes}) != 1
    ):
        raise BundleCompatibilityError(
            "COBANet v1 inputs must share shape ['time', 'batch', channels]"
        )
    input_shape = input_shapes[0]
    tau_gaba = inhibitory[0].get("synapse", {}).get("tau", {})
    if (
        inhibitory[0].get("synapse", {}).get("kind") != "gaba"
        or tau_gaba.get("unit") != "ms"
    ):
        raise BundleCompatibilityError("I→E GABA projection must declare tau in ms")
    if readout_projection[0].get("parameters", []) != [
        f"{readout_projection[0]['id']}.weight"
    ]:
        raise BundleCompatibilityError("readout projection weight is malformed")
    readout_weight = _normal(readout_projection[0], parameters)
    if readout_weight != (5.1, 3.8):
        raise BundleCompatibilityError(
            "current COBANet backend requires readout Normal(5.1, 3.8)"
        )
    readout_tau = readout_projection[0].get("synapse", {}).get("tau", {})
    if readout_tau != {"value": 2.0, "unit": "ms"}:
        raise BundleCompatibilityError(
            "current COBANet backend requires a 2 ms MeanVoltage integrator"
        )

    recurrent_rows = [recurrent_exc[0], inhibitory[0]]
    if w_ee_row is not None:
        recurrent_rows.append(w_ee_row)
    if w_ii_row is not None:
        recurrent_rows.append(w_ii_row)
    sparsity = {_normal_details(row, parameters)[1:] for row in recurrent_rows}
    if len(sparsity) != 1:
        raise BundleCompatibilityError(
            "legacy COBANet execution requires one shared recurrent sparsity policy"
        )
    recurrent_fraction, recurrent_zeroing = sparsity.pop()

    # The graph owns explicit counters. Resolve their physical durations at its
    # declared timestep instead of substituting the legacy simulator defaults.
    refractory = []
    for neuron in (e_neuron, i_neuron):
        count = neuron.get("refractory_steps", 0)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise BundleCompatibilityError(
                "legacy refractory_steps must be a positive integer"
            )
        duration = count * float(dt["value"])
        refractory_steps(duration, float(dt["value"]), policy="exact")
        refractory.append(duration)

    return LegacySettings(
        dt=float(dt["value"]),
        refractory_e_ms=refractory[0],
        refractory_i_ms=refractory[1],
        hidden_size=int(e_pop["size"]),
        input_size=int(input_shape[-1]),
        output_size=int(analogue[0]["size"]),
        w_in=_normal(input_projection[0], parameters),
        w_in_i=_normal(input_projection_i[0], parameters)
        if input_projection_i
        else None,
        w_ee=_normal(w_ee_row, parameters) if w_ee_row else (0.0, 0.0),
        w_ei=_normal(recurrent_exc[0], parameters),
        w_ie=_normal(inhibitory[0], parameters),
        w_ii=_normal(w_ii_row, parameters) if w_ii_row else (0.0, 0.0),
        recurrent_initial_zero_fraction=recurrent_fraction,
        exact_k_initialization=recurrent_zeroing == "exact_k",
        tau_gaba=float(tau_gaba["value"]),
        readout_mode="mem-mean",
    )


def translate_training_v1(
    graph: dict[str, Any], training: dict[str, Any]
) -> TrainingSettings:
    """Recognise the first executable recipe for the COBANet v1 graph subset."""
    graph_parameters = {row["id"] for row in graph.get("parameters", [])}
    projections = graph.get("projections", [])
    recurrent = {
        parameter
        for projection in projections
        if projection.get("connection") == "recurrent"
        for parameter in projection.get("parameters", [])
    }
    trainable_expected = graph_parameters - recurrent

    selected: dict[str, bool] = {}
    trainable_lrs: set[float] = set()
    for group in training.get("parameter_groups", []):
        frozen = bool(group.get("frozen"))
        lr = float(group.get("lr", 0.0))
        if frozen and lr != 0.0:
            raise BundleCompatibilityError(
                f"frozen parameter group {group.get('id')} must use lr 0"
            )
        if not frozen:
            if lr <= 0:
                raise BundleCompatibilityError(
                    f"trainable parameter group {group.get('id')} requires lr > 0"
                )
            trainable_lrs.add(lr)
        for parameter in group.get("parameters", []):
            if parameter in selected:
                raise BundleCompatibilityError(
                    f"training parameter appears in multiple groups: {parameter}"
                )
            selected[parameter] = frozen

    if set(selected) != graph_parameters:
        missing = graph_parameters - set(selected)
        extra = set(selected) - graph_parameters
        detail = []
        if missing:
            detail.append("missing " + ", ".join(sorted(missing)))
        if extra:
            detail.append("unknown " + ", ".join(sorted(extra)))
        raise BundleCompatibilityError(
            "training parameter groups must partition graph parameters: "
            + "; ".join(detail)
        )
    actual_trainable = {
        parameter for parameter, frozen in selected.items() if not frozen
    }
    actual_frozen = {parameter for parameter, frozen in selected.items() if frozen}
    if actual_trainable != trainable_expected or actual_frozen != recurrent:
        raise BundleCompatibilityError(
            "current trainer requires input/readout parameters trainable and "
            "recurrent E/I parameters frozen"
        )
    if len(trainable_lrs) != 1:
        raise BundleCompatibilityError(
            "current trainer requires one learning rate across trainable groups"
        )

    objectives = training.get("objectives", [])
    output_signals = {row["signal"] for row in graph.get("outputs", [])}
    if len(objectives) != 1 or objectives[0] != {
        "kind": "cross_entropy",
        "prediction": next(iter(output_signals), None),
        "target": "digit",
        "weight": 1.0,
    }:
        raise BundleCompatibilityError(
            "current trainer requires one unit-weight cross-entropy objective "
            "from the named output to target 'digit'"
        )
    if training.get("regularizers") or training.get("stop_gradients"):
        raise BundleCompatibilityError(
            "regularizers and stop-gradients are not yet supported"
        )
    surrogate = training.get("surrogate") or {"kind": "fast_sigmoid", "slope": 1.0}
    if surrogate.get("kind") != "fast_sigmoid":
        raise BundleCompatibilityError(
            f"current trainer does not support surrogate {surrogate.get('kind')}"
        )
    surrogate_slope = float(surrogate.get("slope", 0.0))
    if surrogate_slope <= 0:
        raise BundleCompatibilityError("fast-sigmoid surrogate slope must be positive")
    dampening = {
        float(population.get("neuron", {}).get("voltage_grad_dampen", 1.0))
        for population in graph.get("populations", [])
        if population.get("spiking")
    }
    if len(dampening) != 1:
        raise BundleCompatibilityError(
            "current trainer requires one voltage-gradient dampening factor across spiking populations"
        )
    duration = training.get("presentation_duration")
    if duration is not None and duration.get("unit") != "ms":
        raise BundleCompatibilityError(
            "current trainer requires presentation duration in milliseconds"
        )
    gradient_clip = training.get("gradient_clip")
    if gradient_clip not in (None, 1, 1.0):
        raise BundleCompatibilityError(
            "current trainer uses a fixed gradient clip of 1.0"
        )
    optimizer = training.get("optimizer", {})
    if optimizer.get("kind") != "adamw":
        raise BundleCompatibilityError("current trainer requires AdamW")
    optimizer_config = optimizer.get("config", {})
    unknown_optimizer = set(optimizer_config) - {"weight_decay"}
    if unknown_optimizer:
        raise BundleCompatibilityError(
            "unsupported AdamW settings: " + ", ".join(sorted(unknown_optimizer))
        )
    epochs = training.get("epochs")
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
        raise BundleCompatibilityError("training epochs must be a positive integer")
    weight_decay = float(optimizer_config.get("weight_decay", 0.0))
    if weight_decay < 0:
        raise BundleCompatibilityError("weight_decay must be non-negative")
    return TrainingSettings(
        lr=trainable_lrs.pop(),
        weight_decay=weight_decay,
        epochs=epochs,
        surrogate_slope=surrogate_slope,
        voltage_grad_dampen=dampening.pop(),
        presentation_duration_ms=(float(duration["value"]) if duration else None),
    )


_MODEL_SPEC_FLAGS = {
    "--model",
    "--n-hidden",
    "--readout",
    "--dt",
    "--refractory-e-ms",
    "--refractory-i-ms",
    "--refractory-policy",
    "--w-in",
    "--w-in-initial-zero-fraction",
    "--ei-strength",
    "--ei-ratio",
    "--w-ei",
    "--w-ie",
    "--w-ee",
    "--w-ii",
    "--recurrent-initial-zero-fraction",
    "--exact-k-initialization",
    "--tau-gaba",
    "--n-in",
    "--dataset",
    "--load-config",
}

_TRAINING_RECIPE_FLAGS = {
    "--lr",
    "--weight-decay",
    "--epochs",
    "--fr-reg-upper-target-hz",
    "--fr-reg-upper-strength",
    "--trainable-w-ee",
    "--trainable-w-ei",
    "--trainable-w-ie",
    "--trainable-w-ii",
    "--surrogate-slope",
    "--v-grad-dampen",
}

_SIMULATION_RECIPE_FLAGS = {
    "--input-rate",
    "--independent-drive",
    "--independent-drive-i",
    "--quenched-drive",
    "--quenched-drive-i",
}


def apply_bundle_to_args(args, argv: list[str]):
    """Apply bundle structure while preserving execution-only CLI overrides."""
    if not getattr(args, "bundle", None):
        return args
    if getattr(args, "executor", "legacy") == "graph":
        explicit = {item.split("=", 1)[0] for item in argv if item.startswith("--")}
        conflicts = sorted(explicit & {
            "--refractory-e-ms", "--refractory-i-ms", "--refractory-policy",
        })
        if conflicts:
            raise BundleCompatibilityError(
                "graph owns refractory settings; remove conflicting flags: "
                + ", ".join(conflicts)
            )
        # Authenticate data now; graph capability checks happen in planning.
        manifest, graph = load_graph_bundle(args.bundle)
        if args.mode == "train":
            explicit = {item.split("=", 1)[0] for item in argv if item.startswith("--")}
            conflicts = sorted(explicit & _TRAINING_RECIPE_FLAGS)
            if conflicts:
                raise BundleCompatibilityError(
                    "bundle owns training settings; remove conflicting flags: "
                    + ", ".join(conflicts)
                )
            recipe = load_training_recipe(args.bundle, manifest, graph)
            args.epochs = int(recipe["epochs"])
        return args
    if args.mode not in {"sim", "train"}:
        raise BundleCompatibilityError("--bundle currently supports sim and train only")
    explicit = {item.split("=", 1)[0] for item in argv if item.startswith("--")}
    owned = _MODEL_SPEC_FLAGS | (
        _TRAINING_RECIPE_FLAGS if args.mode == "train" else set()
    )
    conflicts = sorted(explicit & owned)
    if conflicts:
        raise BundleCompatibilityError(
            "bundle owns model settings; remove conflicting flags: "
            + ", ".join(conflicts)
        )
    manifest, graph = load_graph_bundle(args.bundle)
    simulation = load_simulation_recipe(args.bundle, manifest, graph)
    if simulation:
        conflicts = sorted(explicit & _SIMULATION_RECIPE_FLAGS)
        if conflicts:
            raise BundleCompatibilityError(
                "bundle owns simulation inputs; remove conflicting flags: "
                + ", ".join(conflicts)
            )
        args._simulation_recipe = simulation
    settings = translate_cobanet_v1(graph)
    if args.mode == "train" and (
        settings.output_size != 10 or settings.input_size != 784
    ):
        raise BundleCompatibilityError(
            "first COBANet bundle route is intentionally limited to MNIST "
            "(784 inputs, 10 outputs)"
        )
    args.model = "ping"
    args.n_hidden = [settings.hidden_size]
    args.n_in = settings.input_size
    args.dataset = "mnist"
    args.dt = settings.dt
    args.refractory_e_ms = settings.refractory_e_ms
    args.refractory_i_ms = settings.refractory_i_ms
    args.refractory_policy = "exact"
    args.readout_mode = settings.readout_mode
    args.w_in = list(settings.w_in)
    args.w_in_i = list(settings.w_in_i) if settings.w_in_i else None
    args.w_in_initial_zero_fraction = 0.0
    args.ei_strength = settings.w_ei[0]
    args.ei_ratio = settings.w_ie[0] / settings.w_ei[0]
    args.w_ei = list(settings.w_ei)
    args.w_ie = list(settings.w_ie)
    args.w_ee = list(settings.w_ee)
    args.w_ii = list(settings.w_ii)
    args.recurrent_initial_zero_fraction = settings.recurrent_initial_zero_fraction
    args.exact_k_initialization = settings.exact_k_initialization
    args.tau_gaba = settings.tau_gaba
    if args.mode == "train":
        recipe = load_training_recipe(args.bundle, manifest, graph)
        training = translate_training_v1(graph, recipe)
        args.lr = training.lr
        args.weight_decay = training.weight_decay
        args.epochs = training.epochs
        args.surrogate_slope = training.surrogate_slope
        args.v_grad_dampen = training.voltage_grad_dampen
        if training.presentation_duration_ms is not None and "--t-ms" not in explicit:
            args.t_ms = training.presentation_duration_ms
    return args
