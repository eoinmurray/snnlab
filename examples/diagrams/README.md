# SNNLang diagram gallery

This gallery exercises `lang.diagram(...)` and `viz.render_diagram(...)` for
human inspection. Run it to explore rendering or review changes to diagram projection, layout,
styling, or labels. It renders nine cases without simulating or training them.

From the repository root, with Graphviz `dot` and `neato` installed and available on `PATH`:

```sh
uv run python examples/diagrams/example.py
```

Open `examples/diagrams/results/index.html`. Each case has explicit circuit and training views. Groups always
show their contents; there are no collapsed or selectively expanded views. Each view includes SVG and PNG downloads and its rendering options.
Click an SVG to inspect it at full size.

The `all_box_types`, `two_layer_classifier` and `exp131_coupled` cases also
include a `circuit_weights` view with compact custom names and a notation legend.
Scripts use smaller text in explicit table cells so SVG previews and raster
exports agree without depending on SVG baseline-shift support. Use `lang.diagram(bundle, weight_labels=True)`
to label every weighted projection with a bold italic W and a source-to-target
subscript. To label selected projections with custom symbols, pass a mapping
from projection IDs to `viz.WeightLabel` objects instead:

```python
visual = lang.diagram(bundle, weight_labels={
    "E_to_I": viz.WeightLabel(symbol="J", subscript="E→I", superscript="rec"),
})
```

The symbol identifies the matrix; its subscript identifies the pathway and its
optional superscript can identify a category such as recurrence. Text is literal,
not LaTeX or HTML. Synapse labels remain above matrix names, and weight labels
are disabled by default. Default names use source→target order, independent of
the matrix's storage orientation. A mapping labels only the specified projections.

The first case, `all_box_types`, shows every supported node kind in its training
view: input, spiking/analogue population, operation, output, training group,
objective, and a manually added neutral external encoder. It includes both
trainable and frozen parameter groups and an explicit E/I group boundary.
A two-layer classifier follows it.
The gallery also includes frozen authoring graphs from Pinglab's exp124
(COBA and PING classifiers), exp131 (coupled PING and phase-response probes),
exp133 (coupled PING and one-way sender–receiver), and exp136 (two senders
converging on a receiver, with separate neuronal cue populations).
The exp136 case exercises inputs inside five authored groups. Its two cue–sender
branches occupy separate rows and converge on the receiver to their right.
For fully grouped input/population circuits with acyclic connections between
groups, layout-only links keep each downstream group's input beyond its upstream
population. Local E/I columns keep I below E; these groups use compact spacing
instead of the wide gaps needed by several independent external inputs.
Inspect labels, group boundaries, arrow directions and routing, overlaps, and
training annotations. This is a review aid, not an automated visual pass/fail test.
Connections use only horizontal and vertical segments with right-angle bends.
Labels are placed separately in clear space around wires and nodes.
Long-range feedback rises into overhead lanes spaced 32 points apart; node
attachment points are separated by at least 18 points.
Frozen training links retain arrowheads and use grey dashed lines.
Coupled circuits are arranged horizontally. Reciprocal E/I pairs
share a column with I below E, including networks without authored groups.
Existing rendering assertions remain in `tests/lang/test_snnlang.py`.

1. Add future cases as factories in `example.py` and include them in `cases()`.
2. Generated files go to `examples/diagrams/results/`. As with the other
   examples, each run replaces the output directory.
3. Override the destination or PNG scale with
   `--out-dir /tmp/snnlang-review --scale 1`. The specified directory is replaced
   on each run. SVG output is independent of scale.
4. If baseline comparisons are introduced later, store approved references in a
   separate `fixtures/` directory. Never regenerate approved references implicitly.

## Pinglab authoring inputs

`inputs/*.json` are diagram inputs, not expected rendered images or simulation
results. They retain the original graph, optional training recipe, and manifest.
Each includes the Pinglab source path, Git revision, and source SHA-256; the hash
identifies the captured file even when the source checkout has uncommitted edits.
Rendering requires the Graphviz `dot` and `neato` executables. `dot` lays out
the nodes and orthogonal wires; a label-clearance pass moves labels into free
space; `neato -n2` exports those fixed positions and routes without a new layout.
No Pinglab imports, custom numerical implementations, datasets, or simulation
execution are needed to render them. The gallery constructs data-only `Bundle`
objects directly: these inputs are not executable experiment bundles.

The snapshots were captured by calling the recipes' authoring functions:

1. exp124: `author_network(configuration(), model)` for `coba` and `ping`.
2. exp131: `author_network()` and `author_phase_response_network()`.
3. exp133: `recipe.author_network()` and `sender_receiver.author_network()`.
   For the latter, only the authoring function was extracted from the source;
   its constants and `recipe.add_ping` helper were supplied without importing
   the module's unrelated storage and execution helpers.
4. exp136: `recipe.author_network()` only. The guarded CLI was not invoked;
   no stimuli or simulations were run. This captures the uncalibrated network
   proposal, including five inputs, eight populations and seventeen projections.

Refresh these files explicitly when a source architecture changes, keeping the
source metadata alongside it. Semantic regressions using these same inputs live
in `tests/lang/test_diagram_regressions.py` and run normally under pytest.
