# snnviz

`snnviz` owns visual rendering for Pinglab's spiking-network tools. It provides
renderer-neutral recording and structural-diagram contracts, numerical
transforms, deterministic layout helpers, a thin Matplotlib composition layer
for activity stills and animations, and Graphviz export for diagrams. It does
not run a simulation or perform scientific analysis.

The public API is intentionally small: compositions retain direct access to
Matplotlib, while `Recording` and the reusable transforms keep visual rendering
grounded in explicit source data. `FigureGrid` is the default composition
system for new multi-panel figures and animation frames. It uses explicit named
placement rather than a textual area map:

```python
grid = snnviz.FigureGrid(
    rows=(0.3, 1.0, 1.4),
    columns=4,
    row_gap=(0.02, 0.05),
)
grid.place("header", row=0, column=0, colspan=4)
grid.place("weights", row=1, column=0)
grid.place("rates", row=1, column=1)
grid.place("spectrum", row=2, column=0, colspan=2)
grid.place("raster", row=2, column=2, colspan=2)

figure = grid.figure(figsize=(6.9, 5.1), dpi=150)
axes = {
    name: grid.add_axes(figure, name)
    for name in ("weights", "rates", "spectrum", "raster")
}
```

Rows run top-to-bottom and columns left-to-right. Tracks use relative weights;
regions can span tracks, nest another `FigureGrid`, or be reserved so no axis
can enter them. Gaps may be one shared value or one value per boundary. Geometry
is resolved once in normalized figure coordinates, so the same layout is stable
across SVG, PDF, PNG posters and every video frame. New grids use the opaque
white, hard-edged, monospace `snnviz` house style by default.

Structural producers lower their own semantics into `Diagram`, `DiagramNode`,
`DiagramEdge`, and `DiagramGroup`; `snnviz.render_diagram` owns deterministic
DOT generation and SVG, PNG, PDF, or DOT export. This keeps source-specific
schema interpretation outside the renderer. For example:

```python
from snnlab import lang as snn
from tools import snnviz

bundle = snn.compile(network)
visual = snn.diagram(bundle, view="circuit", expand_groups={"cell"})
snnviz.render_diagram(visual, "circuit.svg")
```

Structural diagrams default to a height-to-width layout target of `0.58`,
following EXP099 Figure 2's approximately 3:2 article composition. Graphviz
arranges the graph before rendering; text and boxes are never stretched.
The final bounding box also includes titles and margins, so the ratio is a
layout target rather than an exact export size. Pass
`height_to_width_ratio=None` for unconstrained natural layout, or another
positive ratio for a deliberately different composition.

The default diagram house style follows the lab's EXP099 input-map precedent:
opaque white canvas, hard rectangular geometry, bold uppercase monospace text,
generous spacing, black excitatory or ordinary flow, deep-red inhibitory flow,
and amber or cyan only for distinct output or training roles. Diagram labels
name the scientific mechanism rather than internal renderer details. Authored
component membership becomes visible clustering; related populations occupy a
shared target column and use short local names, while mixed computational groups
retain their internal signal-flow order. `DiagramGroup.same_row` lays members
out in their declared order, keeping recurrent arrows from reversing that order;
`same_rank` remains available for deliberately vertical groups. The two options
are mutually exclusive. Ungrouped external inputs without incoming edges align
in one entry column.
The shared `DiagramTheme` typography uses a 26-point title, 16-point node and
cluster labels, and 13-point secondary and edge labels. All text is explicitly
bold Menlo; secondary text uses full-contrast ink. Long card and edge labels
wrap at 14 characters, with 1.2 line spacing inside cards and six-point gaps
between text roles. Wrapping preserves the complete display text and avoids
making the whole graph wider merely to fit one long label.

These sizes are calibrated against EXP099 Figure 2 **at equal displayed
width**, not by comparing source font-size numbers alone. At 720 pixels wide,
EXP099's generated diagram has approximately 19.6-pixel title, 12.1-pixel labels
and 9.8-pixel secondary text, compared with 19.5, 11.7 and 9.4 pixels in Figure 2.
For new graph shapes, inspect exports at the intended article width: increasing
font sizes can also increase Graphviz's bounding box and leave displayed text
no larger. Wrap labels or simplify the declared view before accepting small
text. Do not stretch SVGs or remove scientific content to meet a ratio. SNNLang places related populations vertically using
`same_rank`; explicitly authored `same_row` groups retain horizontal ordering.

## Scientific visual style guide

This guide consolidates the visual conventions already used across Pinglab.
Some rules are enforced by `snnviz` contracts; others are composition rules that
renderers and experiment presentation stages must apply deliberately.

### Current technical reference

These values describe the implementations currently in the repository. The
shared theme values are defaults for new work. Values under "observed
precedents" document existing renderers and are not additional requirements.

#### Colour

| Role | Shared scientific theme | Canvas-animation precedent |
| --- | --- | --- |
| Background | required white `#ffffff` | legacy warm paper `#f3efe6` (nonconforming) |
| Primary ink / E / first series | `#1a1a1a` | `#20201e` |
| Contrast / I / second series | `#c8102e` | `#a62a24` |
| Additional category 1 | cyan `#00b4d8` | — |
| Additional category 2 | amber `#e89400` | — |
| Related/context series | `#3a3a3a`, `#6a6a6a`, `#a0a0a0` | `#77726a` |
| Rules and boundaries | `#e7e5df`, `#d9d5c8` | renderer-specific warm greys |

The Matplotlib categorical cycle is black, red, cyan, amber, dark grey, then
mid grey. The default scalar image map runs from white through red to black.
Every plot, figure, schematic, poster, and video frame must use an opaque pure
white `#ffffff` background. Do not use warm paper, tinted, grey, transparent, or
theme-dependent backgrounds in final visual outputs. The former warm
`snnviz.styles.Theme` background has been removed; its defaults now implement
the shared scientific palette directly.

#### Typography

The shared font stack is JetBrains Mono, IBM Plex Mono, Menlo, Consolas,
Courier New, then DejaVu Sans Mono. Screen figures use the following point-size
ladder:

| Element | Screen | Paper |
| --- | ---: | ---: |
| Base text | 10.5 pt | 8 pt |
| Title | 11 pt, bold and left-aligned | 8.5 pt |
| Axis label | 10 pt | 8 pt |
| Tick label | 9 pt | 7 pt |
| Legend | 9 pt | 7 pt |
| Annotation | 8 pt | 6.5 pt |
| Caption/footer | 8 pt | 6.5 pt |

The EXP099 canvas animation is an observed large-format exception: its main
heading is 17 pt, time readout 12.5 pt, panel headings approximately
8.8–10.8 pt, ordinary annotations approximately 7.2–8.4 pt, and its smallest
legend text 6 pt.

#### Lines, axes, and uncertainty

- Screen-theme axes are 1.4 pt on all four sides; major ticks are 1 pt wide and
  4 pt long, pointing inward. The default data line is 2 pt with butt caps and
  miter joins.
- Paper mode reduces axes and tick widths to 0.8 pt, tick length to 3 pt, data
  lines to 1.3 pt, and patch edges to 0.8 pt.
- The available grid is black at 0.4 pt and `0.15` alpha, drawn below data; it is
  off by default. Current plots commonly use 1.0–1.6 pt for primary traces and
  0.5–0.8 pt dotted or dashed lines for thresholds, chance levels, zero lines,
  and other references.
- Existing uncertainty bands commonly use the series colour at `0.14`–`0.18`
  alpha with no boundary line. This is an observed convention, not a substitute
  for naming the uncertainty statistic.
- Existing spike rasters normally use `|` markers, approximately 0.5–2 points²
  marker area (`scatter(..., s=...)`), and 0.35–0.5 pt strokes. E spikes are
  black and I spikes red. Dense rasters are rasterised rather than forced into
  large SVG files.

#### Figure geometry and output

- Common existing single-figure canvases are 5.6 × 3.15 in, 6.5 in wide, or
  6.9 × 3.88 in. The last is the established full-print-width 16:9 canvas.
- Existing larger 16:9 analysis canvases include 8 × 4.5 in and 12 × 6.75 in.
  Multi-panel height varies with the number and information density of panels;
  it is not fixed merely to preserve 16:9.
- The EXP099 animation uses a compact 14.4 × 7.2 in canvas at 120 DPI during
  composition, with a 240-DPI PNG poster. Its shape follows its wide network and
  six-panel diagnostic composition rather than a global output ratio.
- `experiments.helpers.figsave.save_figure` emits SVG and PDF by default. Dense
  rasters commonly request PNG and PDF instead; schematic line art, sparse
  plots, and ordinary curves remain SVG and PDF.
- The shared screen profile specifies tight bounding boxes and 240 DPI; paper
  mode specifies 300 DPI, 0.02 in padding, and PDF/PS font type 42 for editable
  text. Older and experiment-specific renderers still contain explicit 150- and
  220-DPI exports; those are existing precedents, not the target for new shared
  output.

#### Animation

- `save_animation` uses Matplotlib's FFmpeg writer at 25 frames per second and
  3800 kbps by default, with `blit=False` and a frame interval of `1000 / fps`
  milliseconds.
- `FrameTimeline.sample` selects inclusive source steps uniformly and requires
  `0 < frames <= steps`. `FrameTimeline.compose` accepts inclusive
  `(start_step, end_step, frame_count)` segments; equal endpoints create holds
  and reversed endpoints play backward.
- The current production animation uses 600 frames at the default 25 FPS and
  3800 kbps. It displays a PNG poster alongside its MP4 presentation file.
- Snnviz does not mandate an aspect ratio, physical canvas size or player-control
  band. Choose the canvas around the figure's composition and intended display.
  Animation frames must retain one stable geometry throughout the encoded video;
  validate control overlays for the intended player when they matter.

1. **Render retained evidence truthfully.** Build every visual from an explicit
   recording, analysed measurement, or declared theoretical construction. Treat
   missing signals, inconsistent timelines, and unsupported comparisons as
   errors. Do not invent activity, silently substitute zeros, rerun a simulation,
   or calculate a new scientific estimator while rendering.

2. **Separate evidence, measurement, and appearance.** Compute creates primary
   evidence, analyse derives scientific measurements, and present composes plots,
   figures, posters, and videos. A change to an estimator belongs in analyse; a
   change to visual encoding or layout belongs in present.

3. **Make the scientific question visually primary.** A figure should reveal
   the comparison, mechanism, transition, or uncertainty needed to answer its
   question. Remove decoration that does not help the reader recover that
   relationship. Use an overview-to-detail sequence when a mechanism needs
   context: structure or input, then activity, then aggregate measurement.

4. **Keep encodings semantically stable.** Preserve the same colour, marker,
   line style, ordering, axis meaning, and units for the same entity across
   related panels and frames. Existing Pinglab plots normally use near-black for
   E, PING, or the first/control series and deep red for I, COBA, or the
   second/contrast series. Resolve any collision explicitly and state the mapping
   in the legend or caption. Never change an encoding merely to make a panel look
   more varied.

5. **Use a white background and the restrained Pinglab palette.** Every final
   visual must have an opaque pure white `#ffffff` background, including plots,
   schematics, posters, and every animation frame. The standard scientific plot
   cycle is near-black `#1a1a1a`, deep red `#c8102e`, electric cyan `#00b4d8`,
   amber `#e89400`, then dark and mid grey. Use black and red for one- or
   two-series comparisons. Reserve cyan and amber for genuinely additional
   categories and greys for related subfamilies or context. Do not use the warm
   background currently defined by `snnviz.styles.Theme` in final output.

6. **Do not rely on colour alone.** Where series may overlap or the distinction
   is scientifically important, reinforce colour with labels, line styles,
   markers, position, or direct annotation. Use opacity for intensity or
   secondary context, not to conceal uncertainty or de-emphasise inconvenient
   observations.

7. **Use disciplined typography and construction.** Prefer monospace type,
   left-aligned titles, concise labels, high contrast, hard-edged lines, and
   limited decoration. Static experiment plots should use the shared size tokens
   and Matplotlib configuration in `experiments.helpers.theme` rather than local
   magic numbers. Use grids only when they materially aid quantitative reading.

8. **Make comparisons direct.** Related panels should share scales, limits,
   sampling, conditions, and visual mappings unless a difference is necessary.
   Clearly mark any changed scale, normalization, zoom, selected example, or
   different population size. Use the same illustrative trial across conditions
   when the scientific comparison requires a paired view.

9. **Show aggregation and uncertainty honestly.** Identify whether a display is
   a single illustrative probe, individual repeats, an aggregate, or a
   theoretical expectation. Show and name the aggregation and uncertainty used.
   Do not let a selected representative frame or trial stand in for population
   evidence.

10. **Treat animation time as a scientific axis.** Display simulation time when
    its interpretation matters. Synchronize all panels to the same source step.
    Use `FrameTimeline` deliberately for uniform sampling, slow motion, repeats,
    holds, or reverse playback; disclose nonuniform pacing. Pacing may clarify an
    event but must not imply altered simulated dynamics. The encoding defaults
    are 25 frames per second and 3800 kbps unless a composition documents another
    choice. Preserve the bottom control-safe area described above so playback
    chrome cannot hide the evidence.

11. **Make every published figure independently readable.** Provide meaningful
    alternative text. Captions should identify variables, conditions, units,
    encodings, aggregation, uncertainty, and whether examples are illustrative
    or reused. Captions explain how to read the display; nearby Results prose
    states what the evidence establishes. Do not duplicate the same finding in
    both.

12. **Produce appropriate publication artifacts.** Prefer vector SVG for the
    browser and vector PDF for manuscripts when the content permits it, generated
    from the same figure recipe. Use the shared paper profile for manuscript
    figures and sufficient resolution for rasterised content. Presentation-stage
    exports must remain flat, reproducible outputs with their provenance recorded
    by the experiment runner.

## Composition boundary

`Scene` deliberately does not hide Matplotlib or impose a universal panel type.
Reusable helpers should encode truthful data contracts, deterministic geometry,
or generally useful transforms. Experiment-specific scientific meaning,
storyboarding, annotations, and final layout belong in the experiment's present
stage. A composition may depart from this guide when the science requires it,
but the departure should be deliberate and locally explained.
