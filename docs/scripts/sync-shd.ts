import { createHash } from 'node:crypto';
import { copyFileSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const repo = fileURLToPath(new URL('../../', import.meta.url));
const run = 'shd-59e3371bb384';
// Published scientific evidence stays pinned when benchmark results are overwritten.
const source = join(repo, 'docs/public/shd-benchmark', run);
const load = (name: string) => JSON.parse(readFileSync(join(source, name), 'utf8'));
const hash = (path: string) => createHash('sha256').update(readFileSync(path)).digest('hex');
const presentation = load('presentation.json');
const configuration = load('configuration.json');
if (load('cloud.json').status !== 'complete' || presentation.run_id !== run) {
  throw new Error('SHD documentation requires the pinned completed run');
}
const renderer = join(source, 'plot.py');
if (hash(renderer) !== presentation.renderer_sha256) {
  throw new Error('SHD figures need regeneration after plot.py changes');
}
for (const [name, expected] of Object.entries({ ...presentation.source_sha256, ...presentation.figures })) {
  if (hash(join(source, name)) !== expected) throw new Error(`SHD artifact changed: ${name}`);
}
const recipe = 'benchmarks/shd/shd_recipe.py';
const pinnedRecipe = join(source, 'shd_recipe.py');
if (hash(pinnedRecipe) !== configuration.source_sha256[recipe]) {
  throw new Error('Retained SHD recipe differs from the uploaded numerical source');
}
copyFileSync(
  join(repo, 'benchmarks/shd/benchmark.mdx'),
  join(repo, 'docs/content/docs/shd-benchmark.mdx'),
);
