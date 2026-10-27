# Documentation site

This site uses Astro static output with Fumadocs React islands. Keep `output: static`, preserve the root documentation routes, and verify both root and GitHub Pages base-path builds. The documentation includes Installation, Quickstart, Training, Inference, PyTorch Integration, a collapsible API Reference section with exactly six module pages (`snnlab.lang`, `snnlab.sim`, `snnlab.extensions`, `snnlab.meanfield`, `snnlab.analysis`, `snnlab.viz`), no module subpages, and Changelog, with content and metadata in `content/docs`. Installation covers only PyPI. The root `/` is the Welcome documentation page, authored in `content/docs/index.mdx` and rendered through the shared documentation route and layout. Keep it first in the documentation navigation, before Installation. Run `bun run types:check` and `bun run build` after site changes.

Changelog is generated from the root `CHANGELOG.md` by `scripts/sync-changelog.ts` before development, type checks and builds. Edit the root source instead of the generated page.
The Astro integration in `scripts/content-reload.mjs` runs content generation before page discovery and installs development source watching. Keep generator/source mappings in the shared `scripts/content-sources.mjs` registry, including generator dependencies. Package scripts should invoke Astro directly instead of repeating generator chains. `bun run sync:content` is the standalone generation command; preview uses the existing build without generation.
Keep Changelog last in the documentation navigation when adding pages.
Use `getSource()` to construct navigation from the current content collections when rendering or generating routes. Do not cache the Fumadocs source or search index at module scope: the development process must pick up new pages and sidebar metadata without a restart. Keep the development content glob in `src/lib/source.ts` so Vite invalidates dependent routes on content changes.

Keep Python examples and API signatures readable: use one argument or parameter per line in multi-argument calls and declarations, trailing commas, consistent indentation and descriptive names. Keep simple single-argument calls compact. Apply this style to the source templates and executable example scripts, then regenerate published pages and downloadable copies; update extraction anchors when formatting changes them. Preserve numerical behaviour when making presentation-only changes.

## Local verification

The first Examples page is Defining Networks (`/snnlang`), containing two parts: Feedforward Network and Recurrent Circuit. Each part follows intro, complete code, diagram order. Edit `../examples/snnlang/example.mdx` and `../examples/snnlang/example.py`; `scripts/sync-snnlang.ts` publishes the page, scripts and diagrams. Run the Python script after topology changes. Use expanded view for the recurrent circuit so the layer groups and inhibitory feedback remain visible.

Only change code and inspect generated output files. Do not restart the development server; rely on hot reload. Do not use computer or browser tools, screenshots of the site, or HTTP requests to check the site. Anomancer checks the site personally; automated site checks take too long. Type checks, builds, script execution and viewing generated PNGs or other output files are permitted.

Quickstart is generated before development, type checks and builds from `../examples/quickstart/example.mdx` and `../examples/quickstart/example.py` by `scripts/sync-quickstart.ts`. Run the Python script to regenerate the PNG in its results/ directory after changing the simulation or plotting. Do not edit generated copies in `content/docs` or `public/quickstart`.

Training is generated from `../examples/training/example.mdx` and `../examples/training/example.py` by `scripts/sync-training.ts` before development, type checks and builds. Run the script to regenerate its curves after changing the example. Edit the source template/script rather than generated copies.
All example scripts also generate `network.png` from the compiled bundle. Keep these diagrams in the introductory section, copy them with the example assets, and regenerate them after topology changes. Diagram rendering requires Graphviz's `dot` executable.

Inference is generated from `../examples/inference/example.mdx` and `../examples/inference/example.py` by `scripts/sync-inference.ts`. It loads `network.bundle` and `trained.checkpoint` from the sibling Training example and never retrains automatically. Regenerate its outputs after changing Training's saved model. Keep Inference immediately after Training in navigation.

PyTorch Integration is generated from `../examples/pytorch/example.mdx` and `../examples/pytorch/example.py` by `scripts/sync-pytorch.ts`. It reuses Training's saved bundle, initializes fresh weights, and runs an external PyTorch loop. Keep it immediately before Animate a Simulation in Examples. Regenerate its plots after changes to the example or Training bundle.

Keep input declarations in `api/lang.mdx` and execution binding fields/rules in `api/sim.mdx`. Do not create a separate Inputs reference page; retain its former routes as redirects.

Customisation is generated from `../examples/customisation/example.mdx` and `../examples/customisation/example.py` by `scripts/sync-customisation.ts`. Keep it immediately after Current-based LIF and before API Reference. Regenerate its figures after changes; preserve the same shared weights for standard/adaptive comparisons. Extensions reference documents callback contracts, units, tensor state and named implementation requirements.

Current-based LIF is generated from `../examples/current-lif/example.mdx` and `../examples/current-lif/example.py` by `scripts/sync-current-lif.ts`. Keep it after Inference and before Customisation. Regenerate its diagram and four-row figure after example changes.


Model Components and Results and Analysis guides have been removed. Keep neuron/synapse, weight, result and analysis reference material on the six module API pages; do not regenerate the removed Neurons guide from `examples/neurons/`.

Every example directory contains one `example.py` and a `results/` directory. Scripts replace `results/` on every run. Read generated assets from `results/` in sync scripts; saved Training models are in `examples/training/results/`. Custom callbacks live in their example.py and must stay importable for bundle reload. Keep executable examples focused on producing their result files; explanatory prose belongs in the documentation and independent verification belongs in tests. Documentation sync scripts extract sections using code anchors rather than tutorial comments.

Example documentation templates live beside their Python sources as `../examples/<name>/example.mdx`, including Hodgkin–Huxley, Mean-field and Animate a Simulation. Sync scripts in `scripts/` publish these templates into `content/docs`; edit the colocated source templates.

Published benchmark documentation lives beside its Python source as `../benchmarks/<name>/benchmark.mdx` for SHD, snnTorch comparison and Streaming. Edit these colocated sources; `scripts/sync-benchmarks.ts` publishes them to the existing `content/docs` routes before development, checks and builds, and template edits trigger hot reload. Generated benchmark pages in `content/docs` are ignored by Git. Keep published evidence pinned in `public/`; synchronization must not replace it from mutable benchmark results or execute benchmarks. `scripts/sync-shd.ts` verifies the pinned SHD artifact hashes before publishing its page.
