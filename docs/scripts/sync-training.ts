import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';

const example = new URL('../../examples/training/', import.meta.url);
const publicDir = new URL('../public/training/', import.meta.url);
const script = readFileSync(new URL('example.py', example), 'utf8');
const template = readFileSync(new URL('example.mdx', example), 'utf8');

function section(start: string, end: string, indent = true) {
  const startIndex = script.indexOf(start);
  const endIndex = script.indexOf(end, startIndex + start.length);
  if (startIndex < 0 || endIndex < 0) throw new Error(`Missing Training section: ${start}`);
  return script.slice(startIndex, endIndex).trimEnd()
    .split('\n').map(line => indent && line.startsWith('    ') ? line.slice(4) : line).join('\n').trim();
}
const replacements: Record<string, string> = {
  SCRIPT: script.trimEnd(),
  PROGRESS_SCRIPT: readFileSync(new URL("../../examples/native-training/example.py", import.meta.url), "utf8").trimEnd(),
  DATASET: section('def make_dataset', '\ndef main', false),
  SETUP: section('net = lang.Network(', 'inputs = net.input('),
  INPUTS: section('inputs = net.input(', 'cells = net.population('),
  NETWORK: section('cells = net.population(', 'net.output('),
  OUTPUTS: section('net.output(', 'objective = lang.CrossEntropy('),
  RECIPE: section('objective = lang.CrossEntropy(', 'bundle = lang.compile('),
  BUNDLE: section('bundle = lang.compile(', 'validation = ValidationSpec('),
  TRAIN: section('validation = ValidationSpec(', 'metrics_path = OUTPUT_DIR / "metrics.json"'),
  METRICS: section('metrics_path = OUTPUT_DIR / "metrics.json"', 'grid = viz.FigureGrid('),
};
writeFileSync(new URL('../content/docs/training.mdx', import.meta.url),
  template.replace(/\{\{(\w+)\}\}/g, (_, name) => {
    if (!(name in replacements)) throw new Error(`Unknown Training placeholder: ${name}`);
    return replacements[name];
  }));
mkdirSync(publicDir, { recursive: true });
for (const name of ['example.py', 'training.png', 'network.png']) {
  copyFileSync(new URL(name.endsWith('.py') ? name : `results/${name}`, example), new URL(name, publicDir));
}

const progressDir = new URL("../public/native-training/", import.meta.url);
mkdirSync(progressDir, { recursive: true });
for (const name of ["example.py", "network.png"]) {
  copyFileSync(new URL(`../../examples/native-training/${name.endsWith(".py") ? name : `results/${name}`}`, import.meta.url), new URL(name, progressDir));
}
