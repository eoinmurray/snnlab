import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';

const example = new URL('../../examples/quickstart/', import.meta.url);
const publicDir = new URL('../public/quickstart/', import.meta.url);
const script = readFileSync(new URL('example.py', example), 'utf8');
const template = readFileSync(new URL('example.mdx', example), 'utf8');

function section(start: string, end: string) {
  const startIndex = script.indexOf(start);
  const endIndex = script.indexOf(end, startIndex + start.length);
  if (startIndex < 0 || endIndex < 0) throw new Error(`Missing Quickstart section: ${start}`);
  return script.slice(startIndex, endIndex).trimEnd()
    .split('\n').map(line => line.startsWith('    ') ? line.slice(4) : line).join('\n').trim();
}
const replacements: Record<string, string> = {
  SCRIPT: script.trimEnd(),
  SETUP: section('net = lang.Network(', 'inputs = net.input('),
  INPUTS: section('inputs = net.input(', 'cells = net.population('),
  NETWORK: section('cells = net.population(', 'net.output('),
  OUTPUTS: section('net.output(', 'bundle = lang.compile('),
  BUNDLE: section('bundle = lang.compile(', 'result = simulate('),
  EXECUTION: section('result = simulate(', 'data = result.numpy(batch=0)'),
  RETRIEVAL: section('data = result.numpy(batch=0)', 'grid = viz.FigureGrid('),
};

writeFileSync(new URL('../content/docs/quickstart.mdx', import.meta.url),
  template.replace(/\{\{(\w+)\}\}/g, (_, name) => {
    if (!(name in replacements)) throw new Error(`Unknown Quickstart placeholder: ${name}`);
    return replacements[name];
  }));
mkdirSync(publicDir, { recursive: true });
for (const name of ['example.py', 'quickstart.png', 'network.png']) {
  copyFileSync(new URL(name.endsWith('.py') ? name : `results/${name}`, example), new URL(name, publicDir));
}
