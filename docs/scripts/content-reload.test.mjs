import { expect, test } from 'bun:test';
import { EventEmitter } from 'node:events';
import { fileURLToPath } from 'node:url';
import generatedContent, { contentReload, isContentFile } from './content-reload.mjs';
import { sourceWatchPaths, syncScriptFor, syncScripts, syncScriptsFor } from './content-sources.mjs';

const file = relative => fileURLToPath(new URL(relative, import.meta.url));

test('maps authoring sources to generators without watching generated output as a source', () => {
  expect(syncScriptFor(file('../../CHANGELOG.md'))).toBe('sync-changelog.ts');
  expect(syncScriptFor(file('../../examples/pytorch/example.mdx'))).toBe('sync-pytorch.ts');
  expect(syncScriptFor(file('../../examples/meanfield/example.mdx'))).toBe('sync-meanfield.ts');
  expect(syncScriptFor(file('../../examples/animated-neuron/example.mdx'))).toBe('sync-animated-neuron.ts');
  expect(syncScriptFor(file('./quickstart.mdx'))).toBeUndefined();
  expect(syncScriptFor(file('../../examples/meanfield/example.py'))).toBe('sync-meanfield.ts');
  expect(syncScriptFor(file('../../examples/meanfield/results/network.png'))).toBe('sync-meanfield.ts');
  expect(syncScriptFor(file('../../examples/meanfield/results/equations.tex'))).toBe('sync-meanfield.ts');
  expect(syncScriptFor(file('../../examples/quickstart/example.py'))).toBe('sync-quickstart.ts');
  expect(syncScriptFor(file('../../examples/training/results/network.png'))).toBe('sync-training.ts');
  expect(syncScriptFor(file('../../examples/native-training/example.py'))).toBe('sync-training.ts');
  expect(syncScriptFor(file('../../examples/animated-neuron/results/recording.json'))).toBe('sync-animated-neuron.ts');
  expect(syncScriptFor(file('../../examples/streaming/example.mdx'))).toBe('sync-streaming.ts');
  expect(syncScriptFor(file('../../examples/streaming/example.py'))).toBe('sync-streaming.ts');
  expect(syncScriptFor(file('../../examples/streaming/results/network.png'))).toBe('sync-streaming.ts');
  expect(syncScriptFor(file('../../examples/training/results/trained.checkpoint'))).toBeUndefined();
  for (const name of ['shd', 'snntorch-comparison', 'streaming']) {
    expect(syncScriptFor(file(`../../benchmarks/${name}/benchmark.mdx`))).toBe('sync-benchmarks.ts');
    expect(syncScriptFor(file(`../../benchmarks/${name}/results/report.json`))).toBeUndefined();
  }
  expect(syncScriptFor(file('./sync-benchmarks.ts'))).toBe('sync-benchmarks.ts');
  expect(syncScriptFor(file('./sync-shd.ts'))).toBe('sync-benchmarks.ts');
  expect(syncScriptsFor(file('./sync-content.ts'))).toEqual(syncScripts);
  expect(sourceWatchPaths).toContain(file('../../examples/streaming'));
  expect(syncScriptFor(file('../../benchmarks/scikit-digits/benchmark.mdx'))).toBeUndefined();
  expect(syncScriptFor(file('../content/docs/streaming-benchmark.mdx'))).toBeUndefined();
  expect(syncScriptFor(file('../content/docs/quickstart.mdx'))).toBeUndefined();
  expect(isContentFile(file('../content/docs/api/sim/meta.json'))).toBe(true);
  expect(isContentFile(file('../../src/snnlab/sim/README.md'))).toBe(false);
});

test('Astro generates content before dev, builds and checks, but preview uses the built site', () => {
  for (const command of ['dev', 'build', 'sync']) {
    const calls = [];
    generatedContent(() => calls.push('generate')).hooks['astro:config:setup']({
      command,
      updateConfig: config => calls.push(config.vite.plugins[0].name),
    });
    expect(calls).toEqual(['generate', 'documentation-content-reload']);
  }
  generatedContent(() => { throw new Error('preview must not generate'); }).hooks['astro:config:setup']({
    command: 'preview',
    updateConfig: () => { throw new Error('preview must not install a watcher'); },
  });
});

test('generation failures stop Astro setup before content discovery', () => {
  const integration = generatedContent(() => { throw new Error('Missing example result'); });
  expect(() => integration.hooks['astro:config:setup']({
    command: 'build',
    updateConfig: () => { throw new Error('must not continue after failed generation'); },
  })).toThrow('Missing example result');
});

test('invalidates route modules and reloads for edits, additions and deletions', async () => {
  const watcher = new EventEmitter();
  watcher.add = () => {};
  const httpServer = new EventEmitter();
  const calls = [];
  let done;
  const reloaded = new Promise(resolve => { done = resolve; });
  const server = {
    watcher, httpServer,
    config: { logger: { error: message => { throw new Error(message); } } },
    environments: {
      ssr: { moduleGraph: { invalidateAll: () => calls.push('ssr') } },
      prerender: {
        moduleGraph: { invalidateAll: () => calls.push('prerender') },
        runner: { evaluatedModules: {
          idToModuleMap: new Map([['route', {}]]),
          invalidateModule: () => calls.push('evaluated-route'),
        } },
      },
      client: { moduleGraph: { invalidateAll: () => calls.push('client') } },
    },
    ws: { send: message => { calls.push(message.type); done(); } },
  };
  contentReload().configureServer(server);
  watcher.emit('change', file('../content/docs/installation.mdx'));
  watcher.emit('add', file('../content/docs/new-page.mdx'));
  watcher.emit('unlink', file('../content/docs/removed-page.mdx'));
  await reloaded;
  expect(calls).toEqual(['ssr', 'prerender', 'evaluated-route', 'full-reload']);
  httpServer.emit('close');
  expect(watcher.listenerCount('change')).toBe(0);
});

test('hot reload batches source edits and runs a shared generator once', async () => {
  const watcher = new EventEmitter();
  watcher.add = () => {};
  const httpServer = new EventEmitter();
  const generated = [];
  let done;
  const reloaded = new Promise(resolve => { done = resolve; });
  contentReload(scripts => generated.push(scripts)).configureServer({
    watcher, httpServer,
    config: { logger: { error: message => { throw new Error(message); } } },
    environments: {},
    ws: { send: () => done() },
  });
  watcher.emit('change', file('../../examples/training/example.py'));
  watcher.emit('change', file('../../examples/native-training/example.py'));
  watcher.emit('change', file('../../examples/streaming/example.mdx'));
  watcher.emit('change', file('./sync-shd.ts'));
  watcher.emit('change', file('../../benchmarks/shd/benchmark.mdx'));
  await reloaded;
  expect(generated).toEqual([['sync-training.ts', 'sync-streaming.ts', 'sync-benchmarks.ts']]);
  httpServer.emit('close');
});
