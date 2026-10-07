# Documentation site

This site uses Astro static output with Fumadocs React islands. Keep `output: static`, preserve the root documentation routes, and verify both root and GitHub Pages base-path builds. The documentation includes Installation, Quickstart, Training, Inference, PyTorch Integration, a collapsible API Reference section with Lang, Sim and Viz subgroups and Changelog, with content and metadata in `content/docs`. Installation covers only PyPI. The root is a landing page with basic library copy and links to documentation and GitHub; documentation begins at `/installation`. Run `bun run types:check` and `bun run build` after site changes.

Changelog is generated from the root `CHANGELOG.md` by `scripts/sync-changelog.ts` before development, type checks and builds. Edit the root source instead of the generated page.
Keep Changelog last in the documentation navigation when adding pages.
Use `getSource()` to construct navigation from the current content collections when rendering or generating routes. Do not cache the Fumadocs source or search index at module scope: the development process must pick up new pages and sidebar metadata without a restart. Keep the development content glob in `src/lib/source.ts` so Vite invalidates dependent routes on content changes.

## Local verification

Only change code and inspect generated output files. Do not restart the development server; rely on hot reload. Do not use computer or browser tools, screenshots of the site, or HTTP requests to check the site. Anomancer checks the site personally; automated site checks take too long. Type checks, builds, script execution and viewing generated PNGs or other output files are permitted.

Quickstart is generated before development, type checks and builds from `scripts/quickstart.mdx` and `../examples/quickstart/quickstart.py` by `scripts/sync-quickstart.ts`. Run the Python script to regenerate the PNG beside it after changing the simulation or plotting. Do not edit generated copies in `content/docs` or `public/quickstart`.

Training is generated from `scripts/training.mdx` and `../examples/training/training.py` by `scripts/sync-training.ts` before development, type checks and builds. Run the script to regenerate its curves after changing the example. Edit the source template/script rather than generated copies.
All example scripts also generate `network.png` from the compiled bundle. Keep these diagrams in the introductory section, copy them with the example assets, and regenerate them after topology changes. Diagram rendering requires Graphviz's `dot` executable.

Inference is generated from `scripts/inference.mdx` and `../examples/inference/inference.py` by `scripts/sync-inference.ts`. It loads `network.bundle` and `trained.checkpoint` from the sibling Training example and never retrains automatically. Regenerate its outputs after changing Training's saved model. Keep Inference immediately after Training in navigation.

PyTorch Integration is generated from `scripts/pytorch-integration.mdx` and `../examples/pytorch/training.py` by `scripts/sync-pytorch.ts`. It reuses Training's saved bundle, initializes fresh weights, and runs an external PyTorch loop. Keep it immediately after Inference. Regenerate its plots after changes to the example or Training bundle.

Keep input declarations in `api/lang/network.mdx` and execution binding fields/rules in `api/sim/execution-spec.mdx`. Do not create a separate Inputs reference page; retain its former routes as redirects.

Customisation is generated from `scripts/customisation.mdx` and `../examples/customisation/customisation.py` by `scripts/sync-customisation.ts`. Keep it immediately after Current-based LIF and before API Reference. Regenerate its figures after changes; preserve the same shared weights for standard/adaptive comparisons. Extensions reference documents callback contracts, units, tensor state and named implementation requirements.

Current-based LIF is generated from `scripts/current-lif.mdx` and `../examples/current-lif/current_lif.py` by `scripts/sync-current-lif.ts`. Keep it after PyTorch Integration and before Customisation. Regenerate its diagram and four-row figure after example changes.

Neurons is generated from `scripts/neurons.mdx` and `../examples/neurons/neurons.py` by `scripts/sync-neurons.ts`. Edit those sources and regenerate both COBA and CUBA diagrams and comparison plots after example changes. Keep Neurons, Synapses and Weights together after Customisation; Synapses and Weights are authored directly in `content/docs`.
