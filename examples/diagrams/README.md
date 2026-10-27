# SNNLang diagram gallery

This gallery exercises `lang.diagram(...)` and `viz.render_diagram(...)` for
human inspection. Run it to explore rendering or review changes to diagram projection, layout,
styling, or labels. It renders ten cases without simulating or training them.

From the repository root, with Graphviz `dot` and `neato` installed and available on `PATH`:

```sh
uv run python examples/diagrams/example.py
```

Open `examples/diagrams/results/index.html`. Each case has explicit circuit and training views. Groups always
show their contents; there are no collapsed or selectively expanded views. Each view includes SVG and PNG downloads and its rendering options.
Click an SVG to inspect it at full size.

Connections are unlabelled by default, including their synapse types.
The `all_box_types` and `exp155_cued_digits` cases also include `circuit_labels`
views with user-supplied text and mathematics. Pass a mapping from projection IDs
to strings; omitted projections stay unlabelled:

```python
visual = lang.diagram(
    bundle,
    connection_labels={
        "input_projection": "encoded input",
        "E_to_I": r"$J_{E\to I}$",
        "I_to_E": r"$\frac{g_I}{\tau_I}$",
    },
)
viz.render_diagram(visual, "network.svg")
```

Labels are presentation settings and do not change the bundle. Plain text is
escaped literally, including underscores. Enclose mathematical expressions in
`$...$`; Matplotlib MathText supports LaTeX-style fractions, Greek letters,
subscripts and superscripts without a TeX installation. This is MathText's
supported subset, not a full LaTeX document. Newlines give multiple label rows.
SVG embeds vector mathematics; PNG and PDF embed high-resolution math images.
Labels participate in wire clearance in every format. Raw `diagram_to_dot`
retains mathematical source as literal text; `render_diagram(..., "network.dot")`
writes Graphviz image labels and a `network.labels/` asset directory.

The first case, `all_box_types`, shows every supported node kind in its training
view: input, spiking/analogue population, operation, output, training group,
objective, and a manually added neutral external encoder. It includes both
trainable and frozen parameter groups and an explicit E/I group boundary.
A two-layer classifier follows it.
The gallery also includes frozen authoring graphs from Pinglab's exp124
(COBA and PING classifiers), exp131 (coupled PING and phase-response probes),
exp133 (coupled PING and one-way sender–receiver), and exp136 (two senders
converging on a receiver, with separate neuronal cue populations), and exp155
(two digit/cue senders converging on a third E/I module and a linear classifier).
The exp155 snapshot uses four separate input ports (64 pixels and 16 cue
afferents per sender), one C E spike-count reduction, and the classifier output.
Its feedforward training plan learns digit-input, sender-to-C, and classifier
weights while freezing E/I recurrence and cues. It renders without importing
Pinglab or executing its custom weight initializer.
Ungrouped E/I fan-in circuits with readouts use a topology-derived grid: local pairs stay adjacent, diagnostic reductions share their population's
row, and a reduction feeding both a diagnostic and a classifier gets a separate
diagnostic row. Graphviz routes the wires around these fixed node positions.
Other topologies retain the usual automatic layout.
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
5. exp155: `author_network("learned", configuration())`, using the paired,
   pixel-preserving architecture and the default feedforward training plan. This
   captures four inputs, six populations, twelve projections, two operations and
   one output. The recipe and custom
   initializer source hashes are retained; no datasets, parameter initialization,
   training or simulation were run.

Refresh these files explicitly when a source architecture changes, keeping the
source metadata alongside it. Semantic regressions using these same inputs live
in `tests/lang/test_diagram_regressions.py` and run normally under pytest.
