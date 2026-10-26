import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { createDevSource } from '../src/lib/dev-source';

test('the API reference exposes exactly six module pages without subpages', () => {
  const contents: Record<string, string> = {};
  for (const file of new Bun.Glob('**/*.{md,mdx,json}').scanSync('content/docs')) {
    contents[`/content/docs/${file}`] = readFileSync(`content/docs/${file}`, 'utf8');
  }
  expect(Array.from(new Bun.Glob('**/*.mdx').scanSync('content/docs/api')).sort()).toEqual([
    'analysis.mdx', 'extensions.mdx', 'lang.mdx', 'meanfield.mdx', 'sim.mdx', 'viz.mdx',
  ]);
  const source = createDevSource(contents, {}, '/');
  for (const name of ['lang', 'sim', 'extensions', 'meanfield', 'analysis', 'viz']) {
    expect(source.getPage(['api', name])?.data.title).toBe(`snnlab.${name}`);
    expect(JSON.stringify(source.getPageTree())).toContain(`"url":"/api/${name}"`);
  }
  expect(source.getPage(['api', 'sim', 'graph-executor'])).toBeUndefined();
  expect(JSON.stringify(source.getPageTree())).not.toContain('/api/sim/graph-executor');
});

test('new snapshots reflect page additions, metadata changes, content edits and deletions', () => {
  const key = '/content/docs/api/sim/new-page.mdx';
  const contents: Record<string, string> = {
    '/content/docs/meta.json': '{"pages":["api"]}',
    '/content/docs/api/meta.json': '{"pages":["sim"]}',
    '/content/docs/api/sim/meta.json': '{"title":"Sim","pages":[]}',
  };
  const before = createDevSource(contents, {}, '/');
  expect(before.getPage(['api', 'sim', 'new-page'])).toBeUndefined();
  contents[key] = '---\ntitle: Added page\n---\n\nNew content.';
  contents['/content/docs/api/sim/meta.json'] = '{"title":"Simulation","pages":["new-page"]}';
  const added = createDevSource(contents, {}, '/');
  expect(JSON.stringify(added.getPageTree())).toContain('Added page');
  expect(JSON.stringify(added.getPageTree())).toContain('Simulation');
  expect(added.getPage(['api', 'sim', 'new-page'])?.data.structuredData.contents).toEqual([
    { heading: undefined, content: 'New content.' },
  ]);
  contents[key] = '---\ntitle: Edited page\n---\n\nEdited content.';
  const edited = createDevSource(contents, {}, '/');
  expect(edited.getPage(['api', 'sim', 'new-page'])?.data._raw.body).toContain('Edited content.');
  expect(edited.getPage(['api', 'sim', 'new-page'])?.data.structuredData.contents).toEqual([
    { heading: undefined, content: 'Edited content.' },
  ]);
  delete contents[key];
  expect(createDevSource(contents, {}, '/').getPage(['api', 'sim', 'new-page'])).toBeUndefined();
});
