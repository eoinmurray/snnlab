import { execFileSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const docsDir = fileURLToPath(new URL('../', import.meta.url));
const repoDir = path.dirname(docsDir);

// One registry drives initial generation, source watching and generator dependencies.
export const contentTasks = [
  ...[
    'snnlang', 'excalidraw', 'quickstart', 'training', 'pytorch', 'inference', 'streaming',
    'current-lif', 'customisation', 'hodgkin-huxley', 'meanfield', 'animated-neuron',
  ].map(name => ({
    script: `sync-${name}.ts`,
    examples: name === 'training' ? [name, 'native-training'] : [name],
  })),
  {
    script: 'sync-benchmarks.ts',
    benchmarks: ['shd', 'snntorch-comparison', 'streaming'],
    dependencies: ['sync-shd.ts'],
  },
  { script: 'sync-changelog.ts', files: ['CHANGELOG.md'] },
];

export const syncScripts = contentTasks.map(task => task.script);

export const sourceWatchPaths = [
  path.join(docsDir, 'content/docs'),
  path.join(docsDir, 'scripts'),
  ...contentTasks.flatMap(task => [
    ...(task.examples ?? []).map(name => path.join(repoDir, 'examples', name)),
    ...(task.benchmarks ?? []).map(name => path.join(repoDir, 'benchmarks', name, 'benchmark.mdx')),
    ...(task.files ?? []).map(name => path.join(repoDir, name)),
  ]),
];

export function syncScriptFor(file) {
  const absolute = path.resolve(file);
  const relative = path.relative(docsDir, absolute).split(path.sep).join('/');
  const generator = /^scripts\/(sync-[\w-]+\.ts)$/.exec(relative)?.[1];
  const example = path.relative(path.join(repoDir, 'examples'), absolute).split(path.sep).join('/');
  const exampleSource = /^([^/]+)\/(?:example\.mdx|(?:results\/)?[^/]+\.(?:py|png|json|tex|excalidraw))$/.exec(example)?.[1];
  const benchmark = path.relative(path.join(repoDir, 'benchmarks'), absolute).split(path.sep).join('/');
  const benchmarkSource = /^([^/]+)\/benchmark\.mdx$/.exec(benchmark)?.[1];
  return contentTasks.find(task =>
    generator === task.script || task.dependencies?.includes(generator) ||
    task.examples?.includes(exampleSource) || task.benchmarks?.includes(benchmarkSource) ||
    task.files?.some(name => absolute === path.join(repoDir, name)),
  )?.script;
}

export function syncScriptsFor(file) {
  if (path.resolve(file) === path.join(docsDir, 'scripts/sync-content.ts')) return syncScripts;
  const script = syncScriptFor(file);
  return script ? [script] : [];
}

export function syncContent(scripts = syncScripts) {
  if (!scripts.length) return;
  // Use one Bun process for the batch; a fresh process also avoids cached imports on edits.
  try {
    execFileSync('bun', [path.join(docsDir, 'scripts/sync-content.ts'), ...scripts], {
      cwd: docsDir,
      stdio: 'pipe',
    });
  } catch (error) {
    throw new Error(`Documentation generation failed:\n${error.stderr?.toString() || error.message}`, { cause: error });
  }
}
