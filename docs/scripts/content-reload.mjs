import { execFileSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const docsDir = fileURLToPath(new URL('../', import.meta.url));
const repoDir = path.dirname(docsDir);
const examples = new Map([
  ['quickstart', 'quickstart'], ['training', 'training'],
  ['pytorch', 'pytorch'], ['inference', 'inference'],
  ['current-lif', 'current-lif'], ['customisation', 'customisation'],
]);

export function syncScriptFor(file) {
  const absolute = path.resolve(file);
  if (absolute === path.join(repoDir, 'CHANGELOG.md')) return 'sync-changelog.ts';
  const relative = path.relative(docsDir, absolute).split(path.sep).join('/');
  const template = /^scripts\/(quickstart|training|pytorch-integration|inference|current-lif|customisation)\.mdx$/.exec(relative);
  if (template) return `sync-${template[1] === 'pytorch-integration' ? 'pytorch' : template[1]}.ts`;
  const generator = /^scripts\/(sync-[\w-]+\.ts)$/.exec(relative);
  if (generator && ['sync-changelog.ts', ...[...examples.values()].map(name => `sync-${name}.ts`)].includes(generator[1])) {
    return generator[1];
  }
  const example = path.relative(path.join(repoDir, 'examples'), absolute).split(path.sep).join('/');
  const asset = /^([^/]+)\/[^/]+\.(py|png)$/.exec(example);
  if (asset && examples.has(asset[1])) return `sync-${examples.get(asset[1])}.ts`;
}

export function isContentFile(file) {
  const relative = path.relative(path.join(docsDir, 'content/docs'), path.resolve(file));
  return !relative.startsWith('..') && !path.isAbsolute(relative) && /\.(md|mdx|json)$/.test(relative);
}

export default function contentReload() {
  return {
    name: 'documentation-content-reload',
    apply: 'serve',
    configureServer(server) {
      const logger = server.config.logger;
      const pending = new Set();
      let timer;
      let running = false;
      let closed = false;

      async function flush() {
        if (running || closed) return;
        running = true;
        const files = [...pending];
        pending.clear();
        try {
          const scripts = new Set(files.map(syncScriptFor).filter(Boolean));
          for (const script of scripts) {
            execFileSync('bun', [path.join(docsDir, 'scripts', script)], { cwd: docsDir, stdio: 'pipe' });
          }
          // Vite reattaches this watcher after a configuration reload.
          for (const [name, environment] of Object.entries(server.environments)) {
            if (name === 'client') continue;
            environment.moduleGraph.invalidateAll();
            const evaluated = environment.runner?.evaluatedModules;
            if (evaluated) {
              for (const module of evaluated.idToModuleMap.values()) evaluated.invalidateModule(module);
            }
            environment.hot?.send('astro:content-changed', {});
          }
          server.ws.send({ type: 'full-reload' });
        } catch (error) {
          logger.error(`Content reload failed: ${error.message}`);
        } finally {
          running = false;
          if (pending.size && !closed) timer = setTimeout(flush, 100);
        }
      }

      function changed(file) {
        if (!isContentFile(file) && !syncScriptFor(file)) return;
        pending.add(file);
        clearTimeout(timer);
        timer = setTimeout(flush, 100);
      }

      server.watcher.add([
        path.join(docsDir, 'content/docs'), path.join(docsDir, 'scripts'),
        path.join(repoDir, 'CHANGELOG.md'),
        ...[...examples.keys()].map(name => path.join(repoDir, 'examples', name)),
      ]);
      for (const event of ['add', 'change', 'unlink']) server.watcher.on(event, changed);
      server.httpServer?.once('close', () => {
        closed = true;
        clearTimeout(timer);
        for (const event of ['add', 'change', 'unlink']) server.watcher.off(event, changed);
      });
    },
  };
}
