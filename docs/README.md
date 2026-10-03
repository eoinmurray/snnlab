# snnlab documentation

Fumadocs + Next.js static documentation with KaTeX and browser-side search. Requires Node.js 22 or newer.

```sh
npm ci
npm run dev
```

The documentation overview opens at `/`; guides use paths such as `/quickstart/`. Cloudflare redirects previous `/docs/` links to the new paths.

Author pages in `content/docs/` and order them in `content/docs/meta.json`.

```sh
npm run types:check
npm run build
npm run preview
```

The build exports plain HTML, CSS, JavaScript, search data, and KaTeX fonts into `out/`. Production builds use webpack. No Next.js server is needed. Markdown downloads use explicit `/llms.mdx/docs/.../content.md` routes; runtime content-negotiation rewrites are unavailable on static hosts.

## GitHub Pages

`.github/workflows/docs.yml` checks and builds docs on pull requests and deploys `main` through GitHub Pages Actions. The GitHub build sets `NEXT_PUBLIC_BASE_PATH=/snnlab` and `SITE_URL=https://eoinmurray.github.io/snnlab/`.

## Cloudflare

`wrangler.jsonc` deploys the root-path static export as Workers Static Assets at `ssnlab.eoinmurray.info`. An authenticated Wrangler installation is required.

```sh
SITE_URL=https://ssnlab.eoinmurray.info npm run deploy:cloudflare
```

Cloudflare deployments are manual through Wrangler. GitHub Pages updates automatically from `main`. Never deploy a build made with `/snnlab` as its base path to the custom-domain root.

`source.config.ts` configures remark-math and rehype-katex. Commit `package-lock.json`; `.next/`, `.source/`, `out/`, and `node_modules/` are generated and ignored.
