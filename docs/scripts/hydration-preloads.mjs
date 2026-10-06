import { readdir, readFile, writeFile } from 'node:fs/promises';

export function addHydrationPreloads(html, chunks, base = '/') {
  const prefix = `${base.replace(/\/$/, '')}/`;
  const files = new Set();
  function visit(file) {
    if (files.has(file) || !chunks.has(file)) return;
    files.add(file);
    // Search and other lazy features remain outside the initial download.
    for (const dependency of chunks.get(file)) visit(dependency);
  }

  for (const [island] of html.matchAll(/<astro-island\b[^>]*>/g)) {
    if (!/\bclient="load"/.test(island)) continue;
    for (const [, url] of island.matchAll(/(?:component|renderer)-url="([^"]+)"/g)) {
      if (url.startsWith(prefix)) visit(url.slice(prefix.length));
    }
  }
  if (!files.size) return html;
  const existing = new Set([...html.matchAll(/<link\b[^>]*rel="modulepreload"[^>]*href="([^"]+)"[^>]*>/g)]
    .map((match) => match[1]));
  const links = [...files].map((file) => `${prefix}${file}`)
    .filter((url) => !existing.has(url))
    .map((url) => `<link rel="modulepreload" crossorigin href="${url}">`).join('');
  return html.replace('</head>', `${links}</head>`);
}

export default function hydrationPreloads() {
  const chunks = new Map();
  let base;
  return {
    name: 'documentation-hydration-preloads',
    hooks: {
      'astro:config:done': ({ config }) => { base = config.base; },
      'astro:build:setup': ({ updateConfig }) => {
        updateConfig({ plugins: [{
          name: 'collect-hydration-chunks',
          applyToEnvironment: (environment) => environment.name === 'client',
          generateBundle(_options, bundle) {
            chunks.clear();
            for (const entry of Object.values(bundle)) {
              if (entry.type === 'chunk') chunks.set(entry.fileName, entry.imports);
            }
          },
        }] });
      },
      'astro:build:done': async ({ dir }) => {
        for (const file of await readdir(dir, { recursive: true })) {
          if (!file.endsWith('.html')) continue;
          const url = new URL(file, dir);
          const html = await readFile(url, 'utf8');
          const updated = addHydrationPreloads(html, chunks, base);
          if (updated !== html) await writeFile(url, updated);
        }
      },
    },
  };
}
