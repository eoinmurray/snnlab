# snnlab documentation

Fumadocs + Astro static documentation with KaTeX and browser-side search. Requires Bun 1.4.2 or newer.

```sh
bun install --frozen-lockfile
bun run dev
```

The development server defaults to `http://localhost:3001`. Static preview also uses port 3001.

The site contains Installation, Quickstart, Training, Inference, PyTorch Integration, a collapsible API Reference section and Changelog. `/` is the Welcome documentation page, with the shared sidebar, search and page navigation. Its source is `content/docs/index.mdx`; Installation remains at `/installation`. Page URLs omit trailing slashes: Astro exports `installation.html`, which GitHub Pages serves at `/installation`. Former `/installation/` URLs use static HTML redirects to the slash-free URL. Cloudflare redirects previous `/docs/` links to the new paths.


The library comparison guide lives in `content/docs/comparisons.mdx`. It explains `snnlab.lang` through a runnable circuit and compares SpikingJelly, snnTorch, Brian 2, NEST and Norse using official sources. Its graph diagram lives in `public/comparisons/network.png`; regenerate it when changing the example's topology.

Edit Installation in `content/docs/` and keep navigation in `content/docs/meta.json`. Changelog is generated from the root `CHANGELOG.md` before development, type checks and builds; edit that source file.

Installation and Quickstart stay at the top of the sidebar. Examples opens by
default; API Reference and Benchmarks start collapsed. Comparison with Other SNN Libraries is a top-level page. The navigation folders contain only `meta.json` files and refer
to existing pages using relative paths, preserving their URLs and generated
content locations. Short navigation labels live in `src/lib/navigation.ts`;
sidebar labels and article titles use title case, preserving API and library names. Keep Changelog last.

During development, navigation and page rendering use Vite's current content
files rather than a cached collection snapshot. Edits to the root Changelog, example templates, Python
scripts, or generated PNGs rerun the corresponding sync script automatically.
Python simulations still need to be run explicitly to regenerate their figures.

```sh
bun run types:check
bun run build
bun run preview
```

The build exports plain HTML, CSS, JavaScript, search data, and KaTeX fonts into `out/`. Astro uses `output: static`; no application server is needed. Markdown downloads use explicit `/llms.mdx/docs/.../content.md` routes; runtime content-negotiation rewrites are unavailable on static hosts.

Layout styles are inlined into the HTML to avoid a blocking stylesheet request on
the first visit. KaTeX styles remain a shared external asset, loaded only on pages
whose Markdown contains math delimiters. Search data is parsed when the search
index is requested, rather than while constructing navigation for every page;
development still constructs a fresh source from the current content snapshot.

Documentation links use normal browser navigation. Each documentation page preloads
its React island, renderer and shared JavaScript in the head so sidebar controls can
hydrate promptly. Search stays lazy; the Welcome page uses the same React documentation layout as other pages.

## GitHub Pages

`.github/workflows/docs.yml` checks and builds docs on pull requests and deploys relevant pushes to `main` through GitHub Pages Actions. The production build sets `SITE_BASE_PATH=/` and `SITE_URL=https://snnlab.eoinmurray.info` so documentation is served at the custom-domain root.

Set the repository's GitHub Pages custom domain to `snnlab.eoinmurray.info`. In Cloudflare DNS, use a DNS-only CNAME record named `snnlab` pointing to `eoinmurray.github.io` (without `/snnlab`). GitHub Actions deployments use the Pages repository setting; a `CNAME` file in the export is ignored. Wrangler configures Workers deployments, not this DNS alias.

## Legacy Cloudflare Worker

`wrangler.jsonc` deploys the root-path static export as Workers Static Assets at `ssnlab.eoinmurray.info`. Bun runs Wrangler through `bunx`; Cloudflare authentication is required.

```sh
SITE_URL=https://ssnlab.eoinmurray.info bun run deploy:cloudflare
```

The existing Worker at the separate hostname `ssnlab.eoinmurray.info` is deployed manually through Wrangler. The production hostname `snnlab.eoinmurray.info` points to GitHub Pages and updates automatically from relevant pushes to `main`. Never deploy a build made with `/snnlab` as its base path to a custom-domain root.

`astro.config.mjs` configures remark-math and rehype-katex. Commit `bun.lock`; `.astro/`, `out/`, and `node_modules/` are generated and ignored.

The first Examples page, Defining Networks (`/snnlang`), covers SNNLang authoring and rendering in two parts: Feedforward Network and Recurrent Circuit. Its source template is `../examples/snnlang/example.mdx`, with runnable scripts and generated diagrams in `../examples/snnlang/`. Run `../examples/snnlang/example.py` after topology changes; `scripts/sync-snnlang.ts` publishes the page, complete code, diagrams and downloads before development, checks and builds. Each part follows intro, code, diagram order.

Quickstart prose lives in `../examples/quickstart/example.mdx`; its script and generated PNGs live in `../examples/quickstart/`. Run the Python script after changing it. `scripts/sync-quickstart.ts` publishes the complete script and copies its images and download before development, checks and builds.

Quickstart, Training and Inference are generated from their source templates (`example.mdx`) alongside their Python scripts (`example.py`) in `../examples/<name>/`. Run the corresponding example script to regenerate plots, then use the Bun check/build commands to synchronize downloadable scripts and figures.

Inference follows Training in navigation; PyTorch Integration sits immediately before Animate a Simulation. Its source template is `../examples/inference/example.mdx` and its runnable script is `../examples/inference/example.py`; `scripts/sync-inference.ts` copies the generated diagram, response figure and script into `public/inference/`. Run Training first, then Inference: the latter reads the sibling Training bundle and checkpoint without modifying them.

PyTorch Integration is generated by `scripts/sync-pytorch.ts` from `../examples/pytorch/example.mdx` and `../examples/pytorch/example.py`. It reuses the Training bundle with an external PyTorch loop; its trained state dictionary and metrics are generated locally and excluded from source distributions.

API Reference contains six module pages with no module subpages: `snnlab.lang`, `snnlab.sim`, `snnlab.extensions`, `snnlab.meanfield`, `snnlab.analysis` and `snnlab.viz`. Neurons, synapses, weights, results and analysis are documented in those references. Previous API routes redirect to the corresponding module pages.

Input declarations live in Network; binding types and compatibility rules live in Execution Options. Both former Inputs routes redirect to that reference.

Current-based LIF follows Inference and demonstrates built-in current neurons and synapses. `scripts/sync-current-lif.ts` publishes its script and figures from `../examples/current-lif/`.

Customisation follows Current-based LIF and demonstrates current-based LIF neurons, adaptive dynamics and a custom initializer supplied by an external importable `custom_neurons.py` module. `scripts/sync-customisation.ts` publishes both Python files and figures; run `../examples/customisation/customisation.py` to regenerate outputs, then add `--simulate` to load the saved bundle in a fresh process without registration or rebuilding.

The SHD workload report is authored in `content/docs/shd-benchmark.mdx`. Its four
figures and raw/derived downloads are pinned to `shd-59e3371bb384`; regenerate with
`../benchmarks/shd/plot.py` before running `scripts/sync-shd.ts`. The sync script
verifies artifact, renderer and recipe hashes and runs before development/checks/builds.
It does not download the raw trace, run training or silently choose another run.

The Hodgkin–Huxley example follows Custom Neurons and Synapses in Examples. Edit `../examples/hodgkin-huxley/example.mdx` and `../examples/hodgkin-huxley/example.py`. Run the Python script to regenerate the network diagram, response plot, SciPy error plot and `comparison.json`; `scripts/sync-hodgkin-huxley.ts` publishes the page, measured table and downloads before development, checks and builds.
