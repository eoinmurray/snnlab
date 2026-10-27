import path from 'node:path';
import { docsDir, sourceWatchPaths, syncContent, syncScriptsFor } from './content-sources.mjs';

export function isContentFile(file) {
  const relative = path.relative(path.join(docsDir, 'content/docs'), path.resolve(file));
  return !relative.startsWith('..') && !path.isAbsolute(relative) && /\.(md|mdx|json)$/.test(relative);
}

export function contentReload(generate = syncContent) {
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
          const scripts = new Set(files.flatMap(syncScriptsFor));
          generate([...scripts]);
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
        if (!isContentFile(file) && !syncScriptsFor(file).length) return;
        pending.add(file);
        clearTimeout(timer);
        timer = setTimeout(flush, 100);
      }

      server.watcher.add(sourceWatchPaths);
      for (const event of ['add', 'change', 'unlink']) server.watcher.on(event, changed);
      server.httpServer?.once('close', () => {
        closed = true;
        clearTimeout(timer);
        for (const event of ['add', 'change', 'unlink']) server.watcher.off(event, changed);
      });
    },
  };
}

export default function generatedContent(generate = syncContent) {
  return {
    name: 'documentation-generated-content',
    hooks: {
      'astro:config:setup': ({ command, updateConfig }) => {
        if (command === 'preview') return;
        // Generate before Astro discovers pages, including during `astro check`/`sync`.
        generate();
        updateConfig({ vite: { plugins: [contentReload(generate)] } });
      },
    },
  };
}
