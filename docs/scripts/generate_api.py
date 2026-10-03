"""Generate source-derived public API references without importing simulation modules."""

from __future__ import annotations

import argparse
import ast
import html
import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "docs/content/docs/api"
NOTES = json.loads(Path(__file__).with_name("api_notes.json").read_text())
GROUPS = ("lang", "sim", "viz")
SPECIAL = {
    "__call__",
    "__iter__",
    "__len__",
    "__getitem__",
    "__enter__",
    "__exit__",
    "__getattr__",
    "__rmul__",
}
PARAMETERS = {
    "name": "Name used to identify the authored or rendered object.",
    "id": "Stable identifier in the relevant graph or data contract.",
    "input_id": "Exact declared graph input id.",
    "target_id": "Exact named training target id.",
    "dt_ms": "Simulation timestep in milliseconds.",
    "duration_ms": "Physical presentation duration in milliseconds.",
    "t_ms": "Physical duration in milliseconds.",
    "dt": "Timestep; authoring uses a Quantity and legacy simulation uses milliseconds.",
    "seed": "Seed controlling this operation’s random stream.",
    "order_seed": "Seed for deterministic sample ordering.",
    "device": "Requested or resolved tensor execution device.",
    "executor": "Execution route: legacy or graph; graph must be selected explicitly.",
    "graph": "Serialized graph mapping.",
    "network": "Mutable authoring Network.",
    "net": "Network or model being authored or executed; see this callable’s contract.",
    "plan": "Lowered GraphPlan for the complete graph.",
    "path": "Filesystem source or destination path, as described below.",
    "run_dir": "Directory containing retained execution artifacts.",
    "bundle": "Compiled data bundle or bundle path, as annotated.",
    "input_bindings": "Named dense input bindings, resolved against graph contracts.",
    "event_bindings": "Named sparse event bindings with zero-based step/batch/channel coordinates.",
    "poisson_bindings": "Declared generated Poisson input bindings.",
    "inputs": "Input tensors keyed by graph input id.",
    "targets": "Named integer targets or target objects, according to this contract.",
    "steps_count": "Number of simulation timesteps in an event or Poisson binding.",
    "batch_size": "Number of presentations in a binding or mini-batch.",
    "rates_hz": "Configured input rates in spikes per second.",
    "rate_hz": "Rate in spikes per second.",
    "max_rate_hz": "Maximum rate in spikes per second for encoded input.",
    "tau_ms": "Decay time constant in milliseconds.",
    "recording": "Retained Recording or recording selection, as annotated.",
    "recording_fields": "Explicit field names to retain.",
    "runtime_state": "Dynamic graph state for causal continuation; distinct from training checkpoints.",
    "checkpoint": "Training checkpoint record or authenticated checkpoint path.",
    "training": "Authored training declaration or serialized training recipe.",
    "epochs": "Number of training passes over the selected presentations.",
    "lr": "Learning rate; frozen groups require zero and trainable groups require a positive value.",
    "frozen": "Whether this parameter scope is excluded from optimizer updates.",
    "shape": "Explicit dimensions/axes or layout shape, as required by the containing contract.",
    "unit": "Declared physical unit; projection weights use uS.",
    "signal_type": "Declared signal vocabulary, for example spikes or continuous.",
    "classes": "Number of output/readout classes.",
    "n_e": "Excitatory population size.",
    "n_i": "Inhibitory population size.",
    "synapse": "Serialized synapse specification.",
    "initializer": "Serialized parameter initialization law.",
    "constraint": "Parameter constraint or Graphviz layout constraint, as annotated.",
    "connection": "Declared connection kind, for example feedforward, recurrent or feedback.",
    "delay": "Physical projection delay; graph execution requires integral timestep alignment.",
    "enabled": "Whether an authored projection contributes conductance during execution.",
    "mean": "Parent distribution mean.",
    "std": "Parent distribution standard deviation.",
    "initial_zero_fraction": "Fraction of additional zeroed weights at initialization.",
    "zeroing": "Initialization zeroing strategy: bernoulli or exact_k.",
    "ceiling_hz": "Mean-rate ceiling in spikes per second.",
    "strength": "Regularizer or adaptation scale, according to the containing contract.",
    "frames": "Number of output animation frames.",
    "fps": "Encoded video frames per second.",
    "bitrate": "FFmpeg video bitrate in kilobits per second.",
    "row": "Zero-based grid row, counted from top to bottom.",
    "column": "Zero-based grid column, counted from left to right.",
    "rowspan": "Number of grid rows covered.",
    "colspan": "Number of grid columns covered.",
    "return_provenance": "Whether to include initialization provenance in the result.",
}


def esc(value):
    return (
        html.escape(str(value), quote=False)
        .replace("{", "&#123;")
        .replace("}", "&#125;")
        .replace("$", "&#36;")
    )


def cell(value):
    return "`" + str(value).replace("|", "\\|").replace("\n", " ") + "`"


def expr(node):
    return ast.unparse(node) if node is not None else None


def public(node):
    return isinstance(
        node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    ) and not node.name.startswith("_")


def link(module, name=None):
    group, mod = module.split(".", 1)
    return f"/api/{group}/{mod}" + (f"#{name.lower().replace('.', '')}" if name else "")


def source_link(path, node):
    return f"https://github.com/eoinmurray/snnlab/blob/main/{path.relative_to(ROOT).as_posix()}#L{node.lineno}"


MODULES = {}
for group in GROUPS:
    for path in sorted((ROOT / "src/snnlab" / group).glob("*.py")):
        if path.name.startswith("_"):
            continue
        tree = ast.parse(path.read_text())
        MODULES[f"{group}.{path.stem}"] = (path, tree)


def fields(node, module):
    result = []
    classes = {
        n.name: n for n in MODULES[module][1].body if isinstance(n, ast.ClassDef)
    }
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id in classes:
            result.extend(fields(classes[base.id], module))
    own = []
    for n in node.body:
        if (
            isinstance(n, ast.AnnAssign)
            and isinstance(n.target, ast.Name)
            and not n.target.id.startswith("_")
        ):
            default = expr(n.value)
            if isinstance(n.value, ast.Call) and expr(n.value.func) == "field":
                kw = {k.arg: k.value for k in n.value.keywords}
                default = (
                    expr(kw["default"])
                    if "default" in kw
                    else (
                        f"field(default_factory={expr(kw['default_factory'])})"
                        if "default_factory" in kw
                        else None
                    )
                )
            own.append((n.target.id, expr(n.annotation), default))
    for row in own:
        result = [r for r in result if r[0] != row[0]] + [row]
    return result


def args(node):
    out = []
    positional = node.args.posonlyargs + node.args.args
    defaults = [None] * (len(positional) - len(node.args.defaults)) + list(
        node.args.defaults
    )
    for n, default in zip(positional, defaults):
        if n.arg in ("self", "cls"):
            continue
        out.append((n.arg, expr(n.annotation), expr(default)))
    if node.args.vararg:
        out.append(
            ("*" + node.args.vararg.arg, expr(node.args.vararg.annotation), "variadic")
        )
    for n, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        out.append((n.arg, expr(n.annotation), expr(default)))
    if node.args.kwarg:
        out.append(
            ("**" + node.args.kwarg.arg, expr(node.args.kwarg.annotation), "variadic")
        )
    return out


def signature(node, label=None):
    clone = ast.FunctionDef(
        name=node.name,
        args=node.args,
        body=[ast.Pass()],
        decorator_list=[],
        returns=node.returns,
        type_comment=None,
    )
    value = ast.unparse(ast.fix_missing_locations(clone)).split(":\n")[0]
    return value.replace("def " + node.name, "def " + (label or node.name), 1)


def table(rows, header):
    if not rows:
        return ""
    return (
        "| "
        + " | ".join(header)
        + " |\n| "
        + " | ".join(["---"] * len(header))
        + " |\n"
        + "".join("| " + " | ".join(row) + " |\n" for row in rows)
        + "\n"
    )


def walk_body(node):
    # Include validations in nested helpers while excluding nested class definitions.
    for n in ast.walk(node):
        yield n


def validations(node):
    rows = []
    seen = set()
    for n in walk_body(node):
        if isinstance(n, ast.Raise) and n.exc is not None:
            exception = expr(n.exc)
            if exception not in seen:
                seen.add(exception)
                rows.append([cell(exception)])
    return table(rows, ["Explicit exception expression"])


def prose(node, key):
    result = ""
    if key in NOTES:
        result += esc(NOTES[key]) + "\n\n"
    doc = ast.get_docstring(node)
    if doc:
        # Keep original parameter sections, examples and scientific notation exactly.
        result += "Source docstring:\n\n```text\n" + doc + "\n```\n\n"
    return result


def callable_section(node, key, label, path, level=2):
    text = "#" * level + " " + label + "\n\n"
    text += f"[View source]({source_link(path, node)})\n\n"
    decorators = [expr(n) for n in node.decorator_list]
    if decorators:
        text += "Decorators: " + ", ".join(cell(d) for d in decorators) + ".\n\n"
    text += "```python\n" + signature(node, label) + "\n```\n\n" + prose(node, key)
    rows = args(node)
    text += table(
        [
            [
                cell(n),
                cell(t or "unannotated"),
                cell(d) if d is not None else "required",
                esc(
                    PARAMETERS.get(
                        n.lstrip("*"),
                        "Defined by the source contract and implementation below.",
                    )
                ),
            ]
            for n, t, d in rows
        ],
        ["Parameter", "Annotation", "Default", "Meaning"],
    )
    if node.returns:
        text += "Return annotation: " + cell(expr(node.returns)) + ".\n\n"
    if not isinstance(node, ast.ClassDef):
        returned = []

        def scan(n):
            for child in ast.iter_child_nodes(n):
                if isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    continue
                if isinstance(child, ast.Return):
                    value = expr(child.value) or "None"
                    if value not in returned:
                        returned.append(value)
                scan(child)

        scan(node)
        if returned:
            text += "Return expressions (branch-dependent; names refer to the linked implementation):\n\n"
            text += "\n".join("```python\n" + r + "\n```" for r in returned) + "\n\n"
    validation = validations(node)
    if validation:
        text += (
            "Explicit exceptions in this implementation; called helpers may raise additional errors:\n\n"
            + validation
        )
    # Complete callable body is available for unannotated legacy APIs and sparse docstrings.
    body = ast.get_source_segment(path.read_text(), node)
    text += (
        "<details>\n<summary>Implementation</summary>\n\n```python\n"
        + textwrap.dedent(body)
        + "\n```\n\n</details>\n\n"
    )
    return text


EXPECTED = {}
COVERAGE = {
    "scope": "All top-level non-private definitions, module-owned constants/type aliases, and declared public class members in non-private lang/sim/viz modules. Third-party inherited APIs and underscore-prefixed implementation helpers are excluded. Relevant protocol special methods are included.",
    "modules": {},
    "package_exports": {},
}

for module, (path, tree) in MODULES.items():
    group, mod = module.split(".")
    nodes = [n for n in tree.body if public(n)]
    constants = []
    for n in tree.body:
        targets = (
            n.targets
            if isinstance(n, ast.Assign)
            else [n.target]
            if isinstance(n, ast.AnnAssign)
            else []
        )
        for target in targets:
            if isinstance(target, ast.Name) and not target.id.startswith("_"):
                constants.append(
                    (
                        target.id,
                        expr(n.annotation) if isinstance(n, ast.AnnAssign) else None,
                        expr(n.value),
                        n,
                    )
                )
    text = f"---\ntitle: snnlab.{module}\ndescription: Complete declared API of the {mod} module, with signatures, data fields, validation and source.\n---\n\n"
    text += f"[Back to {group} reference](/api/{group})\n\n"
    text += (
        esc(
            NOTES.get(
                module,
                ast.get_docstring(tree) or f"Module-level API for snnlab.{module}.",
            )
        )
        + "\n\n"
    )
    text += "The signatures, defaults, fields, docstrings and implementation excerpts below are generated from the Python source. Annotations are shown as declared; `unannotated` means the source supplies no type annotation. These pages document callable surfaces, including legacy support utilities, without promising backend support for every declaration.\n\n"
    text += table(
        [
            [
                f"[{n.name}]({link(module, n.name)})",
                "class" if isinstance(n, ast.ClassDef) else "function",
            ]
            for n in nodes
        ],
        ["Symbol", "Kind"],
    )
    coverage = []
    for n in nodes:
        coverage.append(n.name)
        key = module + "." + n.name
        if not isinstance(n, ast.ClassDef):
            text += callable_section(n, key, n.name, path)
            continue
        text += (
            "## "
            + n.name
            + "\n\n"
            + f"[View source]({source_link(path, n)})\n\n"
            + prose(n, key)
        )
        bases = [expr(b) for b in n.bases]
        decorators = [expr(d) for d in n.decorator_list]
        if bases:
            text += (
                "Bases: "
                + ", ".join(cell(b) for b in bases)
                + ". Inherited third-party framework APIs follow their owning library.\n\n"
            )
        if decorators:
            text += (
                "Class decorators: " + ", ".join(cell(d) for d in decorators) + ".\n\n"
            )
        init = next(
            (
                m
                for m in n.body
                if isinstance(m, ast.FunctionDef) and m.name == "__init__"
            ),
            None,
        )
        dataclass = any(d.startswith("dataclass") for d in decorators)
        fs = fields(n, module)
        if init:
            text += (
                "Constructor:\n\n```python\n"
                + signature(init, n.name).removeprefix("def ")
                + "\n```\n\n"
            )
            text += table(
                [
                    [
                        cell(a),
                        cell(t or "unannotated"),
                        cell(d) if d is not None else "required",
                        esc(
                            PARAMETERS.get(
                                a.lstrip("*"),
                                "Defined by the constructor implementation below.",
                            )
                        ),
                    ]
                    for a, t, d in args(init)
                ],
                ["Parameter", "Annotation", "Default", "Meaning"],
            )
            text += prose(init, key + ".__init__")
            coverage.append(n.name + ".__init__")
        elif dataclass:
            ctor = ", ".join(
                a + (": " + t if t else "") + (" = " + d if d is not None else "")
                for a, t, d in fs
            )
            text += (
                "Dataclass constructor parameters. Factory defaults are shown as field declarations; omit these arguments to create fresh values per instance:\n\n```python\n"
                + n.name
                + "("
                + ctor
                + ")\n```\n\n"
            )
        elif not bases:
            text += "No explicit constructor is declared; default object construction takes no arguments.\n\n"
        if fs:
            text += "Declared fields, including fields inherited from local data classes:\n\n"
            text += table(
                [
                    [
                        cell(a),
                        cell(t or "unannotated"),
                        cell(d) if d is not None else "required",
                        esc(
                            PARAMETERS.get(
                                a,
                                "Stored member of this data contract; see the class docstring and serialization methods.",
                            )
                        ),
                    ]
                    for a, t, d in fs
                ],
                ["Field", "Annotation", "Default", "Meaning"],
            )
        attrs = []
        for m in n.body:
            if isinstance(m, ast.Assign):
                for a in m.targets:
                    if isinstance(a, ast.Name) and not a.id.startswith("_"):
                        attrs.append([cell(a.id), cell(expr(m.value))])
        if attrs:
            text += "Class attributes:\n\n" + table(
                attrs, ["Name", "Initial expression"]
            )
        if init:
            instance = []
            for m in ast.walk(init):
                if isinstance(m, (ast.Assign, ast.AnnAssign)):
                    targets = m.targets if isinstance(m, ast.Assign) else [m.target]
                    for a in targets:
                        if (
                            isinstance(a, ast.Attribute)
                            and isinstance(a.value, ast.Name)
                            and a.value.id == "self"
                            and not a.attr.startswith("_")
                            and a.attr not in {x[0] for x in instance}
                        ):
                            instance.append(
                                (
                                    a.attr,
                                    expr(m.annotation)
                                    if isinstance(m, ast.AnnAssign)
                                    else None,
                                    expr(m.value),
                                )
                            )
            if instance:
                text += (
                    "Instance members assigned by the constructor (expressions are evaluated when constructed):\n\n"
                    + table(
                        [
                            [cell(a), cell(t or "unannotated"), cell(v)]
                            for a, t, v in instance
                        ],
                        ["Member", "Annotation", "Initial expression"],
                    )
                )
        ctor_validation = "".join(
            validations(m)
            for m in n.body
            if isinstance(m, ast.FunctionDef)
            and m.name in ("__init__", "__post_init__")
        )
        if ctor_validation:
            text += (
                "Constructor/initialization exception expressions:\n\n"
                + ctor_validation
            )
        for m in n.body:
            if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                not m.name.startswith("_") or m.name in SPECIAL
            ):
                coverage.append(n.name + "." + m.name)
                text += callable_section(
                    m, key + "." + m.name, n.name + "." + m.name, path, 3
                )
        # Include constructor, post-init behavior and dynamic forwarding in exact source.
        text += (
            "<details>\n<summary>Complete class implementation</summary>\n\n```python\n"
            + textwrap.dedent(ast.get_source_segment(path.read_text(), n))
            + "\n```\n\n</details>\n\n"
        )
    if constants:
        text += "## Constants and type aliases\n\nInitial source expressions are shown, not evaluated runtime values. Legacy configuration may mutate module defaults.\n\n"
        text += table(
            [
                [
                    cell(a),
                    cell(t or "unannotated"),
                    cell(v),
                    f"[Source]({source_link(path, n)})",
                ]
                for a, t, v, n in constants
            ],
            ["Name", "Annotation", "Initial expression", "Source"],
        )
    if not nodes and not constants:
        text += "This module declares no non-private top-level API symbols. Its underscore-prefixed implementation helpers are outside this reference scope.\n"
    EXPECTED[DEST / group / (mod + ".mdx")] = text
    COVERAGE["modules"][module] = {
        "symbols": coverage,
        "constants": [a for a, _, _, _ in constants],
    }

INTRO = {
    "lang": "Author graph structure, readouts, training and simulation recipes; compile them into deterministic data bundles. Creating declarations does not simulate or train a network.",
    "sim": "Execute graphs, resolve reproducible inputs, train and resume models, authenticate retained artifacts, and compare numerical outputs. Graph-native typed requests and legacy runners have distinct entry points and defaults.",
    "viz": "Validate retained recordings, project renderer-neutral diagrams, compose deterministic figure layouts, and encode animations. Rendering must consume retained evidence with consistent physical timelines.",
}
for group in GROUPS:
    tree = ast.parse((ROOT / "src/snnlab" / group / "__init__.py").read_text())
    exports = {}
    imported = {}
    for n in tree.body:
        if isinstance(n, ast.ImportFrom) and n.level == 1:
            for a in n.names:
                imported[a.asname or a.name] = (
                    f"{group}.{n.module}" if n.module else f"{group}.{a.name}",
                    a.name,
                    n.module is None,
                )
        if isinstance(n, ast.Assign):
            for target in n.targets:
                if isinstance(target, ast.Name) and target.id == "_EXPORTS":
                    for name, mod in ast.literal_eval(n.value).items():
                        exports[name] = (f"{group}.{mod}", name, False)
                if (
                    isinstance(target, ast.Name)
                    and target.id == "__all__"
                    and group != "viz"
                ):
                    for name in ast.literal_eval(n.value):
                        if name in imported:
                            exports[name] = imported[name]
    if group == "viz":
        exports["__version__"] = (f"{group}._version", "__version__", False)
    text = (
        f"---\ntitle: snnlab.{group}\ndescription: Complete API reference for {group}, including exports and every declared module API.\n---\n\n"
        + INTRO[group]
        + "\n\n"
    )
    text += "## Imports\n\n```python\nfrom snnlab import " + group + "\n```\n\n"
    if group == "sim":
        text += 'The package root exports only `__version__`. Import runtime APIs from their submodules:\n\n```python\nfrom snnlab.sim.execution import ExecutionSpec, simulate, train, infer\nfrom snnlab.sim import models, metrics\n```\n\nSelect `executor="graph"` explicitly in typed graph requests. `snnlab.sim.train.train` is the separate legacy runner; it is not `snnlab.sim.execution.train`.\n\n'
    text += "## Package exports\n\n"
    rows = []
    for name, (module, symbol, is_mod) in sorted(exports.items()):
        if name == "__version__":
            version_tree = ast.parse(
                (ROOT / "src/snnlab" / group / "_version.py").read_text()
            )
            version = next(
                expr(n.value)
                for n in version_tree.body
                if isinstance(n, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id == "__version__" for t in n.targets
                )
            )
            rows.append(
                [
                    cell(name),
                    cell(version),
                    "Component format version; distinct from the combined distribution version.",
                ]
            )
        elif module in MODULES:
            defined = any(
                public(n) and n.name == symbol for n in MODULES[module][1].body
            )
            url = link(module, symbol if defined else None)
            rows.append(
                [cell(name), f"[{module}]({url})", "module" if is_mod else "re-export"]
            )
    text += table(rows, ["Export", "Defined in / value", "Role"])
    text += "## Modules\n\n"
    mods = [m for m in MODULES if m.startswith(group + ".")]
    text += table(
        [
            [
                f"[{m}]({link(m)})",
                esc(NOTES.get(m, ast.get_docstring(MODULES[m][1]) or m).split("\n")[0]),
            ]
            for m in mods
        ],
        ["Module", "Responsibility"],
    )
    text += "## Coverage and conventions\n\nThis reference includes every top-level non-private function and class, module-owned constant/type alias, declared public method/property, constructor and relevant protocol special method in the modules above. Data-class constructor fields include local inheritance. Third-party inherited APIs belong to their own libraries; underscore-prefixed implementation helpers are excluded. Imported standard-library and third-party names are not snnlab APIs.\n\nSource docstrings are reproduced as text to preserve their parameter notes, examples, equations and formatting. Explicit exceptions are extracted from the implementation; indirect exceptions can also propagate. Expand the implementation panels for complete branch behavior, especially where legacy code has no annotations.\n\nRegenerate with `python3.12 docs/scripts/generate_api.py`; `--check` fails if source-derived reference pages drift. Authored explanatory notes live in `docs/scripts/api_notes.json`.\n"
    EXPECTED[DEST / group / "index.mdx"] = text
    EXPECTED[DEST / group / "meta.json"] = (
        json.dumps(
            {
                "title": f"snnlab.{group}",
                "pages": ["index"] + [m.split(".")[1] for m in mods],
            },
            indent=2,
        )
        + "\n"
    )
    COVERAGE["package_exports"][group] = sorted(exports)
EXPECTED[DEST / "index.mdx"] = """---
title: API reference
description: Complete source-derived Python API references for lang, sim and viz.
---

Choose a package to inspect its public exports, modules, functions, classes, methods, properties, signatures, defaults, fields, return expressions and validation behavior.

| Package | Responsibility |
| --- | --- |
| [snnlab.lang](/api/lang) | Author graphs, recipes and deterministic bundles. |
| [snnlab.sim](/api/sim) | Simulate, train, bind inputs, retain artifacts and compare outputs. |
| [snnlab.viz](/api/viz) | Compose recordings, diagrams, figures and animations. |

The reference is generated from source without importing runtime modules or downloading datasets. Hand-authored notes explain numerical and lifecycle contracts. Implementation panels retain the full definitions where docstrings or annotations are sparse.

These are API reference pages, not scientific validation claims. Consult [scientific contracts](/contracts) for units, timing and provenance, and the guides for complete runnable workflows.
"""
EXPECTED[DEST / "meta.json"] = (
    json.dumps(
        {"title": "API reference", "pages": ["index", "lang", "sim", "viz"]}, indent=2
    )
    + "\n"
)
EXPECTED[ROOT / "docs/api-coverage.json"] = json.dumps(COVERAGE, indent=2) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    opts = parser.parse_args()
    stale = []
    for path, content in EXPECTED.items():
        if opts.check:
            if not path.exists() or path.read_text() != content:
                stale.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    for path in DEST.rglob("*"):
        if path.is_file() and path not in EXPECTED:
            if opts.check:
                stale.append(str(path.relative_to(ROOT)))
            else:
                path.unlink()
    symbols = sum(len(v["symbols"]) for v in COVERAGE["modules"].values())
    constants = sum(len(v["constants"]) for v in COVERAGE["modules"].values())
    if stale:
        print("Stale API references:\n" + "\n".join(stale), file=sys.stderr)
        return 1
    print(
        f"{'Checked' if opts.check else 'Generated'} {len(MODULES)} module references, {symbols} symbols/members and {constants} constants/type aliases."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
