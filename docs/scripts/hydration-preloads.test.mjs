import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import hydrationPreloads, { addHydrationPreloads } from './hydration-preloads.mjs';

const chunks = new Map([
  ['_astro/docs.js', ['_astro/shared.js']],
  ['_astro/renderer.js', ['_astro/shared.js']],
  ['_astro/shared.js', ['_astro/docs.js']],
  ['_astro/search.js', []],
]);
const page = (base = '/') => `<html><head></head><body><astro-island client="load" component-url="${base}_astro/docs.js" renderer-url="${base}_astro/renderer.js"></astro-island></body></html>`;

test('preloads the immediate island and shared dependencies once, without lazy chunks', () => {
  const html = addHydrationPreloads(page(), chunks);
  const head = html.slice(0, html.indexOf('</head>'));
  for (const name of ['docs', 'renderer', 'shared']) {
    assert.equal(head.split(`href="/_astro/${name}.js"`).length - 1, 1);
  }
  assert.ok(!html.includes('search.js'));
  assert.equal(addHydrationPreloads(html, chunks), html);
});

test('preserves the GitHub Pages base path', () => {
  const html = addHydrationPreloads(page('/snnlab/'), chunks, '/snnlab');
  assert.ok(html.includes('href="/snnlab/_astro/docs.js"'));
  assert.ok(!html.includes('href="/_astro/'));
});

test('leaves static pages and deferred islands unchanged', () => {
  const landing = '<html><head></head><body>Welcome</body></html>';
  assert.equal(addHydrationPreloads(landing, chunks), landing);
  const deferred = page().replace('client="load"', 'client="visible"');
  assert.equal(addHydrationPreloads(deferred, chunks), deferred);
});

test('collects the client environment during the unified build and updates exported HTML', async () => {
  const integration = hydrationPreloads();
  integration.hooks['astro:config:done']({ config: { base: '/' } });
  let plugin;
  integration.hooks['astro:build:setup']({ target: 'server', updateConfig: (config) => { plugin = config.plugins[0]; } });
  assert.equal(plugin.applyToEnvironment({ name: 'client' }), true);
  assert.equal(plugin.applyToEnvironment({ name: 'prerender' }), false);
  plugin.generateBundle({}, Object.fromEntries([...chunks].map(([fileName, imports]) =>
    [fileName, { type: 'chunk', fileName, imports }])));
  const dir = await mkdtemp(join(tmpdir(), 'snnlab-preloads-'));
  try {
    const file = join(dir, 'index.html');
    await writeFile(file, page());
    await integration.hooks['astro:build:done']({ dir: pathToFileURL(`${dir}/`) });
    assert.equal(await readFile(file, 'utf8'), addHydrationPreloads(page(), chunks));
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
