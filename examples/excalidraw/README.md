# Editable Excalidraw network

This example authors an 80-E/20-I conductance-based circuit with excitatory and
inhibitory connections, exports native editable Excalidraw elements, and renders
the saved drawing to PNG. It does not simulate the circuit.

Install Graphviz (`dot` and `neato`) separately through your operating system
package manager. Install the [official Excalidraw CLI](https://github.com/excalidraw/cli)
separately, for example with `bun install -g @excalidraw/cli`. The `excalidraw`
command must be on `PATH`, and rendering needs an installed Chrome, Chromium,
Edge or Firefox browser. No account, API key or subscription is needed. SNNLab
does not install or update these tools.

From the repository root:

```sh
uv run python examples/excalidraw/example.py
```

The script replaces its `results/` directory on every run, producing
`network.excalidraw`, its CLI-rendered `network.png`, a matching static
`network-static.png`, and the reusable `layout.json` specification.

1. Open `results/network.excalidraw` in the free [Excalidraw web editor](https://excalidraw.com).
2. Move the grouped cards, edit text or adjust arrows, and save the drawing back
   to a local `.excalidraw` file. Arrows are bound to their source and target boxes.
3. Render the saved file without regenerating it from the bundle:

   ```python
   from snnlab import viz

   viz.render_excalidraw(
       "examples/excalidraw/results/network.excalidraw",
       "examples/excalidraw/results/network.png",
       scale=2,
   )
   ```

An agent can edit the same JSON document, render it, inspect the PNG, and repeat.
Keep the edited source with the final image. Exporting again produces a fresh
drawing: existing files are protected unless `overwrite=True` is supplied, and
architecture changes are not merged into hand-edited drawings automatically.
Copy your finished drawing outside `results/` before rerunning this example.

Graphviz provides the initial placement and orthogonal routes; export does not
require the Excalidraw CLI. Boxes, group boundaries, labels and arrows remain
native elements rather than an embedded screenshot. Text is grouped with its
owner, and every semantic element retains SNNLab IDs and the source graph digest
in `customData.snnlab`. Inhibitory connections use a bar arrowhead, and feedback
connections are dashed. Specify mathematical connection labels with `$...$`:

```python
layout = viz.DiagramLayout(
    below={"I": "E"},
    align_y=[("drive", "E", "out:E_spikes")],
    edges={
        "drive_to_E": viz.EdgeLayout(label="External drive"),
        "E_to_I": viz.EdgeLayout(label=r"$J_{E\to I}$"),
        "I_to_E": viz.EdgeLayout(label=r"$\frac{g_I}{\tau_I}$"),
    },
)
drawing = bundle.excalidraw_export(
    "network.excalidraw",
    layout=layout,
)
```

These are illustrative annotations: `J` labels the E-to-I coupling, `g_I` denotes
inhibitory conductance, and `tau_I` denotes its decay timescale. The labels do not
evaluate expressions or alter the circuit's parameters.

MathText typesets label lines containing mathematical expressions. Those lines
become high-resolution transparent PNG images embedded as base64 data URLs in
the document's `files` mapping. Plain lines remain editable text, and newline-
separated mixed text/math labels are supported. No supporting image files or TeX
installation are needed. The supported syntax is Matplotlib MathText's LaTeX
subset, including Greek letters, subscripts and fractions. Invalid mathematics
raises `ValueError` before a destination drawing is written.

Equation images are grouped with their arrows and retain each original source
line in `customData.snnlab.math_source`, plus the complete label and projection
ID. You can move or resize them in the web editor. Changing an expression needs
a regenerated image; the web editor and CLI do not typeset its source themselves.
The initial view uses the same card summaries as the static diagram renderer;
complete node titles, details and badges are retained in element metadata.

Rendering reads only the saved file, requires neither Graphviz nor the original
bundle, and never changes the source. It replaces the PNG only after successful
rendering. `scale` is an integer from 1 to 16; the CLI limits output to 16384
pixels per side and 64 megapixels. Rendering disables optional font downloads;
CJK text can fall back to installed system fonts.

## Placement controls

`viz.DiagramLayout` keeps annotation content and placement together. Use `nodes`
for pinned positions, wrapping, sizes, named ports and annotations; `groups` for
nested containers and row/column/grid arrangements; and `edges` for ports,
waypoints, external feedback lanes and labels. `constraints` supports required
relations and weighted preferences. Unknown IDs, inconsistent constraints,
blocked routes and impossible label placements raise errors.

Coordinates use logical canvas units, with x increasing right and y increasing
down. Positions pin box centres unless a node anchor is supplied. Relative node
positions are measured from the containing group's top-left corner. Export scale
changes resolution without changing these coordinates. Font availability can
change text measurements across machines.

```python
layout = viz.DiagramLayout.read("layout.json")
scene = viz.resolve_diagram(
    lang.diagram(bundle),
    layout=layout,
)
scene.write("geometry.json")
updated = scene.move("I", (400, 700))
viz.render_diagram(updated, "network.svg")
viz.export_excalidraw(updated, "adjusted.excalidraw")
updated.layout.write("adjusted-layout.json")
```

`move` pins the other node centres and reroutes connections, while still enforcing
required constraints. A move that violates them raises an error; adjust the
specification when intentionally changing those relationships. Rendering a saved
`.excalidraw` file continues to use its saved geometry and Canvas frame. Resize
the frame if manual edits extend the drawing beyond it. Browser/MCP edits do not
automatically update a layout sidecar, and re-export does not merge them.
