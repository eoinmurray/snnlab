import { createHash } from 'node:crypto';
import { copyFileSync, mkdirSync, readFileSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const repo = fileURLToPath(new URL('../../', import.meta.url));
const run = 'shd-59e3371bb384';
const source = join(repo, 'benchmarks/shd/results', run);
const destination = join(repo, 'docs/public/shd-benchmark', run);
const load = (name: string) => JSON.parse(readFileSync(join(source, name), 'utf8'));
const hash = (path: string) => createHash('sha256').update(readFileSync(path)).digest('hex');
const presentation = load('presentation.json');
const configuration = load('configuration.json');
if (load('cloud.json').status !== 'complete' || presentation.run_id !== run) {
  throw new Error('SHD documentation requires the pinned completed run');
}
const renderer = join(repo, 'benchmarks/shd/plot.py');
if (hash(renderer) !== presentation.renderer_sha256) {
  throw new Error('SHD figures need regeneration after plot.py changes');
}
for (const [name, expected] of Object.entries({ ...presentation.source_sha256, ...presentation.figures })) {
  if (hash(join(source, name)) !== expected) throw new Error(`SHD artifact changed: ${name}`);
}
const recipe = 'benchmarks/shd/shd_recipe.py';
if (hash(join(repo, recipe)) !== configuration.source_sha256[recipe]) {
  throw new Error('Retained SHD recipe differs from the uploaded numerical source');
}
const files = readdirSync(source).filter((name) => /\.(json|png|pdf|svg)$/.test(name));
files.push(...readdirSync(join(source, 'tracing')).filter((name) => /\.(json|csv)$/.test(name) && name !== 'trace.json').map((name) => `tracing/${name}`));
for (const name of files) {
  const target = join(destination, name);
  mkdirSync(dirname(target), { recursive: true });
  copyFileSync(join(source, name), target);
}
copyFileSync(renderer, join(destination, 'plot.py'));
copyFileSync(join(repo, recipe), join(destination, 'shd_recipe.py'));
