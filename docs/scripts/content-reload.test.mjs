import { expect, test } from 'bun:test';
import { EventEmitter } from 'node:events';
import { fileURLToPath } from 'node:url';
import contentReload, { isContentFile, syncScriptFor } from './content-reload.mjs';

const file = relative => fileURLToPath(new URL(relative, import.meta.url));

test('maps authoring sources to generators without watching generated output as a source', () => {
  expect(syncScriptFor(file('../../CHANGELOG.md'))).toBe('sync-changelog.ts');
  expect(syncScriptFor(file('./pytorch-integration.mdx'))).toBe('sync-pytorch.ts');
  expect(syncScriptFor(file('../../examples/quickstart/quickstart.py'))).toBe('sync-quickstart.ts');
  expect(syncScriptFor(file('../../examples/training/network.png'))).toBe('sync-training.ts');
  expect(syncScriptFor(file('../../examples/training/trained.checkpoint'))).toBeUndefined();
  expect(syncScriptFor(file('../content/docs/quickstart.mdx'))).toBeUndefined();
  expect(isContentFile(file('../content/docs/api/sim/meta.json'))).toBe(true);
  expect(isContentFile(file('../../src/snnlab/sim/README.md'))).toBe(false);
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
