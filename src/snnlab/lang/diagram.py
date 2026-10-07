"""Lower snnlang bundle semantics into renderer-neutral diagrams."""

from __future__ import annotations

from collections.abc import Collection

from snnlab import extensions as E
from snnlab.viz import Diagram, DiagramEdge, DiagramGroup, DiagramNode

from .compiler import Bundle


def diagram(
    bundle: Bundle,
    *,
    view: str = "circuit",
    expand_groups: Collection[str] = (),
) -> Diagram:
    """Project a bundle into a diagram without choosing a rendering backend."""

    if view not in {"circuit", "training", "expanded"}:
        raise ValueError("view must be circuit, training, or expanded")
    graph = bundle.graph
    if len(graph["projections"]) + len(graph["operations"]) > 120 and view != "circuit":
        raise ValueError(
            "graph is too dense for a legible expanded static view; filter or collapse it"
        )
    groups = {group["id"]: group for group in graph["groups"]}
    collapsed = view in {"circuit", "training"}
    expanded_groups = set(expand_groups)
    if expanded_groups and not collapsed:
        raise ValueError(
            "expand_groups is only available for circuit and training views"
        )
    unknown_groups = expanded_groups - groups.keys()
    if unknown_groups:
        raise ValueError(
            "cannot expand unknown groups: " + ", ".join(sorted(unknown_groups))
        )

    member_group = {
        member: group["id"] for group in graph["groups"] for member in group["members"]
    }
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

    for row in graph["inputs"]:
        add_node(
            DiagramNode(
                id=row["id"],
                title=row["id"].replace("_", " "),
                detail=" × ".join(map(str, row["shape"])),
                badge=row["signal_type"],
                kind="input",
                classes=("input",),
                pen_width=1.0,
                margin=(0.16, 0.12),
            )
        )

    if collapsed:
        for group_id, group in sorted(groups.items()):
            if group_id in expanded_groups:
                continue
            populations = [
                pop for pop in graph["populations"] if pop["id"] in group["members"]
            ]
            operations = [
                op for op in graph["operations"] if op["id"] in group["members"]
            ]
            if not populations and not operations:
                continue
            detail = "\n".join(
                [
                    f"{local_title(pop['id'], group_id)} · {pop['size']:,} units"
                    for pop in populations
                ]
                + (
                    [
                        f"{len(operations)} operation{'s' if len(operations) != 1 else ''}"
                    ]
                    if operations
                    else []
                )
            )
            add_node(
                DiagramNode(
                    id=group_id,
                    title=group_id.replace("_", " "),
                    detail=detail,
                    badge="component",
                    kind="component",
                    classes=("component",),
                    pen_width=1.0,
                )
            )

    for population in graph["populations"]:
        if (
            collapsed
            and population.get("group") in groups
            and population.get("group") not in expanded_groups
        ):
            continue
        spiking = population["spiking"]
        title_lower = population["id"].lower()
        inhibitory = (
            title_lower == "i" or title_lower.endswith("_i") or "inhib" in title_lower
        )
        add_node(
            DiagramNode(
                id=population["id"],
                title=(
                    local_title(population["id"], population.get("group"))
                    if view == "expanded" or population.get("group") in expanded_groups
                    else population["id"].replace("_", " ")
                ),
                detail=f"{population['size']:,} units\n"
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

    if view == "expanded":
        for operation in graph["operations"]:
            add_node(
                DiagramNode(
                    id=operation["id"],
                    title=local_title(operation["id"], operation.get("group")),
                    detail=operation["kind"].replace("_", " "),
                    badge="operation",
                    kind="operation",
                    classes=("operation",),
                    pen_width=1.0,
                    margin=(0.14, 0.1),
                )
            )

    visible_groups = (
        groups
        if view == "expanded"
        else {group_id: groups[group_id] for group_id in sorted(expanded_groups)}
    )
    population_ids = {population["id"] for population in graph["populations"]}
    for group_id, group in visible_groups.items():
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
        for signal in ("spikes", "voltage", *E.state_units("neuron", population["neuron"])):
            signal_shapes[f"{population['id']}.{signal}"] = (
                "time",
                "batch",
                population["size"],
            )
    for output in graph["outputs"]:
        add_node(
            DiagramNode(
                id="out:" + output["id"],
                title=output["id"].replace("_", " "),
                detail=" × ".join(map(str, signal_shapes.get(output["signal"], ()))),
                badge="output",
                kind="output",
                accent_role="output_line",
                classes=("output",),
                pen_width=1.0,
                margin=(0.16, 0.12),
            )
        )

    def mapped(owner: str) -> str:
        group = member_group.get(owner)
        return (
            group
            if collapsed and group in rendered and group not in expanded_groups
            else owner
        )

    emitted_edges: set[tuple[str, str, str]] = set()
    for projection in graph["projections"]:
        source = mapped(projection["source"].partition(".")[0])
        target = mapped(projection["target"].partition(".")[0])
        if source not in rendered or target not in rendered:
            continue
        if source == target and collapsed:
            continue
        polarity = projection["polarity"]
        connection = projection["connection"]
        key = (source, target, polarity)
        if key in emitted_edges and view == "circuit":
            continue
        emitted_edges.add(key)
        show_receptor = source == target or connection != "recurrent"
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
                    if view != "circuit" and show_receptor
                    else ""
                ),
                connection=connection,
                id=projection["id"],
                classes=(polarity, connection),
                constraint=connection != "feedback",
                pen_width=1.3 if connection in {"recurrent", "feedback"} else 1.1,
            )
        )

    if collapsed:
        for operation in graph["operations"]:
            target = mapped(operation["id"])
            for source_signal in operation["sources"]:
                source = mapped(source_signal.partition(".")[0])
                key = (source, target, "operation")
                if (
                    source != target
                    and source in rendered
                    and target in rendered
                    and key not in emitted_edges
                ):
                    emitted_edges.add(key)
                    edges.append(
                        DiagramEdge(
                            source=source,
                            target=target,
                            role="signal",
                            label="signal",
                            pen_width=1.0,
                        )
                    )

    if view == "expanded":
        for operation in graph["operations"]:
            for source_signal in operation["sources"]:
                source = source_signal.partition(".")[0]
                if source in rendered:
                    edges.append(
                        DiagramEdge(
                            source=source, target=operation["id"], role="signal"
                        )
                    )
        signal_owner = {
            f"{operation['id']}.value": operation["id"]
            for operation in graph["operations"]
        }
        signal_owner |= {
            f"{population['id']}.voltage": population["id"]
            for population in graph["populations"]
        }
        signal_owner |= {
            f"{population['id']}.spikes": population["id"]
            for population in graph["populations"]
        }
        for output in graph["outputs"]:
            owner = signal_owner.get(
                output["signal"], output["signal"].partition(".")[0]
            )
            if owner in rendered:
                edges.append(
                    DiagramEdge(
                        source=owner,
                        target="out:" + output["id"],
                        role="output",
                        pen_width=1.1,
                    )
                )
    else:
        operation_group = {
            operation["id"]: mapped(operation["id"])
            for operation in graph["operations"]
        }
        for output in graph["outputs"]:
            owner = output["signal"].partition(".")[0]
            owner = operation_group.get(owner, mapped(owner))
            if owner in rendered:
                edges.append(
                    DiagramEdge(
                        source=owner,
                        target="out:" + output["id"],
                        role="output",
                        pen_width=1.1,
                    )
                )

    if view == "training" and bundle.training:
        for parameter_group in bundle.training["parameter_groups"]:
            frozen = parameter_group["frozen"]
            add_node(
                DiagramNode(
                    id="train:" + parameter_group["id"],
                    title=parameter_group["id"].replace("_", " "),
                    detail=(
                        f"{len(parameter_group['parameters'])} tensors · "
                        f"lr {parameter_group['lr']:g}"
                    ),
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
            add_node(
                DiagramNode(
                    id=objective_id,
                    title=objective["kind"].replace("_", " "),
                    detail=objective["target"],
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
        for parameter_group in bundle.training["parameter_groups"]:
            targets = {
                member_group.get(
                    parameter["id"].partition(".")[0],
                    parameter["id"].partition(".")[0],
                )
                for parameter in graph["parameters"]
                if parameter["id"] in parameter_group["parameters"]
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
