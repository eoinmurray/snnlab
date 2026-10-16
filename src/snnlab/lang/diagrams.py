"""Lower authoring bundles into renderer-neutral diagrams."""

from __future__ import annotations

import math
from dataclasses import replace

from snnlab import extensions as E
from snnlab.lang.compilation.bundles import Bundle
from snnlab.viz import Diagram, DiagramEdge, DiagramGroup, DiagramNode


def diagram(
    bundle: Bundle,
    *,
    view: str = "circuit",
) -> Diagram:
    """Project a bundle into a diagram without choosing a rendering backend."""

    if view not in {"circuit", "training", "expanded"}:
        raise ValueError("view must be circuit, training, or expanded")
    graph = bundle.graph
    if len(graph["projections"]) + len(graph["operations"]) > 120:
        raise ValueError(
            "graph is too dense for a legible explicit static view; filter it"
        )
    groups = {group["id"]: group for group in graph["groups"]}
    nodes: list[DiagramNode] = []
    edges: list[DiagramEdge] = []
    diagram_groups: list[DiagramGroup] = []
    rendered: set[str] = set()

    def add_node(node: DiagramNode) -> None:
        nodes.append(node)
        rendered.add(node.id)

    def local_title(identifier: str, group_id: str | None) -> str:
        prefix = f"{group_id}_" if group_id else ""
        return identifier.removeprefix(prefix).replace("_", " ")

    def shape_text(shape) -> str:
        return " × ".join(map(str, shape))

    def unit_line(unit: str | None) -> list[str]:
        return [f"Unit: {unit}"] if unit and unit != "1" else []

    def operation_text(operation) -> str:
        kind = operation["kind"]
        if kind.startswith("reduce_"):
            return kind.removeprefix("reduce_").capitalize() + " over time"
        if kind == "linear":
            return f"Linear projection · {operation['shape'][-1]} outputs"
        if kind == "custom_operation":
            definition = operation.get("config", {}).get(
                "definition", "Custom operation"
            )
            return definition.rsplit("/", 1)[0].rsplit(".", 1)[-1].replace("_", " ")
        return kind.replace("_", " ").capitalize()

    for row in graph["inputs"]:
        add_node(
            DiagramNode(
                id=row["id"],
                title=row["id"].replace("_", " "),
                detail="\n".join(
                    [
                        row["signal_type"].capitalize(),
                        shape_text(row["shape"]),
                        *unit_line(
                            row.get("unit") if row["signal_type"] != "spikes" else None
                        ),
                    ]
                ),
                badge=row["signal_type"],
                kind="input",
                classes=("input",),
                pen_width=1.0,
                margin=(0.16, 0.12),
            )
        )

    for population in graph["populations"]:
        spiking = population["spiking"]
        outgoing = [
            projection["polarity"]
            for projection in graph["projections"]
            if projection["source"].partition(".")[0] == population["id"]
        ]
        inhibitory = bool(outgoing) and all(role == "inhibitory" for role in outgoing)
        add_node(
            DiagramNode(
                id=population["id"],
                title=local_title(population["id"], population.get("group")),
                detail=f"{population['size']:,} {'neurons' if spiking else 'units'}\n"
                + {
                    "coba_lif": "COBA-LIF",
                    "cuba_lif": "CUBA-LIF",
                    "cuba_alif": "CUBA-ALIF",
                    "coba_alif": "COBA-ALIF",
                    "cuba_adex": "CUBA-AdEx",
                    "coba_adex": "COBA-AdEx",
                    "lif": "LIF",
                    "leaky_integrator": "Leaky integrator",
                }.get(
                    population["neuron"]["kind"],
                    population["neuron"]["kind"].replace("_", " "),
                ),
                badge="spiking" if spiking else "analogue",
                kind="population",
                accent_role="inhibitory" if inhibitory else "ink",
                classes=("population", "spiking" if spiking else "analogue"),
            )
        )

        if "tau_mem" in population["neuron"]:
            tau = population["neuron"]["tau_mem"]
            node = nodes[-1]
            nodes[-1] = replace(
                node,
                detail=node.detail + f"\nMembrane τ: {tau['value']:g} {tau['unit']}",
            )

    for operation in graph["operations"]:
        add_node(
            DiagramNode(
                id=operation["id"],
                title=local_title(operation["id"], operation.get("group")),
                detail="\n".join(
                    [
                        operation_text(operation),
                        shape_text(operation["shape"]),
                        *unit_line(operation.get("unit")),
                    ]
                ),
                badge="operation",
                kind="operation",
                classes=("operation",),
                pen_width=1.0,
                margin=(0.14, 0.1),
            )
        )

    population_ids = {population["id"] for population in graph["populations"]}
    for group_id, group in groups.items():
        members = tuple(member for member in group["members"] if member in rendered)
        if members:
            diagram_groups.append(
                DiagramGroup(
                    id=group_id,
                    label=group_id.replace("_", " "),
                    members=members,
                    same_rank=len(members) > 1
                    and all(member in population_ids for member in members),
                )
            )

    signal_shapes = {f"{row['id']}.value": row["shape"] for row in graph["operations"]}
    for population in graph["populations"]:
        for signal in (
            "spikes",
            "voltage",
            *E.state_units("neuron", population["neuron"]),
        ):
            signal_shapes[f"{population['id']}.{signal}"] = (
                "time",
                "batch",
                population["size"],
            )
    signals = {
        f"{row['id']}.value": row for row in graph["inputs"] + graph["operations"]
    }
    predictions = {
        objective["prediction"]
        for objective in (bundle.training or {}).get("objectives", [])
        if objective["kind"] == "cross_entropy"
    }

    def output_quantity(signal: str, name: str) -> tuple[str, str | None]:
        row = signals.get(signal, {})
        unit = row.get("unit")
        if signal in predictions or name == "class_scores":
            return "Class scores", unit
        if signal.endswith(".voltage") or unit == "mV":
            return "Mean voltage" if row.get(
                "kind"
            ) == "reduce_mean" else "Voltage", unit or "mV"
        if unit == "Hz":
            return "Firing rate", unit
        if signal.endswith(".spikes") or row.get("signal_type") == "spikes":
            return "Spikes", unit or "spike"
        if unit == "spike" and row.get("kind") == "reduce_sum":
            return "Spike count", unit
        if row.get("signal_type") == "mask":
            return "Mask", unit
        return "Continuous signal" if row.get(
            "signal_type"
        ) == "continuous" else "Signal", unit

    for output in graph["outputs"]:
        quantity, unit = output_quantity(output["signal"], output["id"])
        add_node(
            DiagramNode(
                id="out:" + output["id"],
                title=output["id"].replace("_", " "),
                detail="\n".join(
                    [
                        quantity,
                        shape_text(
                            signal_shapes.get(
                                output["signal"],
                                signals.get(output["signal"], {}).get("shape", ()),
                            )
                        ),
                        *unit_line(unit),
                    ]
                ),
                badge="output",
                kind="output",
                accent_role="output_line",
                classes=("output",),
                pen_width=1.0,
                margin=(0.16, 0.12),
            )
        )

    for projection in graph["projections"]:
        source = projection["source"].partition(".")[0]
        target = projection["target"].partition(".")[0]
        if source not in rendered or target not in rendered:
            continue
        polarity = projection["polarity"]
        connection = projection["connection"]
        edges.append(
            DiagramEdge(
                source=source,
                target=target,
                role=polarity,
                label=(
                    {"ampa": "AMPA", "gaba": "GABA"}.get(
                        projection["synapse"]["kind"],
                        projection["synapse"]["kind"].replace("_", " "),
                    )
                ),
                connection=connection,
                id=projection["id"],
                classes=(polarity, connection),
                constraint=connection != "feedback",
                pen_width=1.3 if connection in {"recurrent", "feedback"} else 1.1,
            )
        )

    for operation in graph["operations"]:
        for source_signal in operation["sources"]:
            source = source_signal.partition(".")[0]
            if source in rendered:
                edges.append(
                    DiagramEdge(source=source, target=operation["id"], role="signal")
                )
    for output in graph["outputs"]:
        owner = output["signal"].partition(".")[0]
        if owner in rendered:
            edges.append(
                DiagramEdge(source=owner, target="out:" + output["id"], role="output")
            )

    if view == "training" and bundle.training:
        parameter_sizes = {
            row["id"]: math.prod(row["shape"]) for row in graph["parameters"]
        }
        for parameter_group in bundle.training["parameter_groups"]:
            frozen = parameter_group["frozen"]
            count = sum(
                parameter_sizes[identifier]
                for identifier in parameter_group["parameters"]
            )
            info = ["Frozen" if frozen else "Trainable"]
            if not frozen:
                info.append(f"Learning rate: {parameter_group['lr']:g}")
            info.append(f"{count:,} parameters")
            add_node(
                DiagramNode(
                    id="train:" + parameter_group["id"],
                    title=parameter_group["id"].replace("_", " "),
                    detail="\n".join(info),
                    badge="frozen" if frozen else "trainable",
                    kind="training",
                    accent_role="training_line",
                    classes=("training", "frozen" if frozen else "trainable"),
                    pen_width=1.0,
                    margin=(0.15, 0.1),
                )
            )
        for index, objective in enumerate(bundle.training["objectives"]):
            objective_id = f"objective:{index}"
            definition = objective.get("definition") or objective.get("config", {}).get(
                "definition"
            )
            title = (
                definition.rsplit("/", 1)[0].rsplit(".", 1)[-1]
                if definition
                else objective["kind"]
            )
            info = [f"Target: {objective['target']}"]
            if objective.get("weight", 1) != 1:
                info.append(f"Weight: {objective['weight']:g}")
            settings = objective.get("config", {})
            for key, value in sorted(settings.items()):
                if key != "definition" and isinstance(value, (str, int, float, bool)):
                    info.append(f"{key.replace('_', ' ').capitalize()}: {value}")
                    break
            add_node(
                DiagramNode(
                    id=objective_id,
                    title=title[:1].upper() + title[1:].replace("_", " "),
                    detail="\n".join(info),
                    badge="objective",
                    kind="objective",
                    accent_role="output_line",
                    classes=("objective",),
                    pen_width=1.0,
                    margin=(0.15, 0.1),
                )
            )
            matching = next(
                (
                    output["id"]
                    for output in graph["outputs"]
                    if output["signal"] == objective["prediction"]
                ),
                None,
            )
            if matching:
                edges.append(
                    DiagramEdge(
                        source="out:" + matching,
                        target=objective_id,
                        role="output",
                        pen_width=1.1,
                    )
                )
        parameter_owners: dict[str, set[str]] = {}
        for projection in graph["projections"]:
            for parameter in projection["parameters"]:
                parameter_owners.setdefault(parameter, set()).add(
                    projection["target"].partition(".")[0]
                )
        for operation in graph["operations"]:
            for parameter in operation["parameters"]:
                parameter_owners.setdefault(parameter, set()).add(operation["id"])
        for parameter_group in bundle.training["parameter_groups"]:
            targets = {
                owner
                for parameter in parameter_group["parameters"]
                for owner in parameter_owners.get(parameter, ())
            }
            for target in sorted(targets):
                if target in rendered:
                    edges.append(
                        DiagramEdge(
                            source="train:" + parameter_group["id"],
                            target=target,
                            role="training",
                            constraint=False,
                            frozen=parameter_group["frozen"],
                        )
                    )

    return Diagram(
        name="snnlang",
        nodes=tuple(nodes),
        edges=tuple(edges),
        groups=tuple(diagram_groups),
        title=graph["name"].replace("_", " ") + " network",
        metadata={"view": view, "source": "snnlang"},
    )
