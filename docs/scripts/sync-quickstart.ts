import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';

const example = new URL('../../examples/quickstart/', import.meta.url);
const publicDir = new URL('../public/quickstart/', import.meta.url);
const script = readFileSync(new URL('example.py', example), 'utf8');
const template = readFileSync(new URL('quickstart.mdx', import.meta.url), 'utf8');

function section(start: string, end: string) {
  const startIndex = script.indexOf(start);
  const endIndex = script.indexOf(end, startIndex + start.length);
  if (startIndex < 0 || endIndex < 0) throw new Error(`Missing Quickstart section: ${start}`);
  return script.slice(startIndex, endIndex).trimEnd()
    .split('\n').map(line => line.startsWith('    ') ? line.slice(4) : line).join('\n').trim();
}
const replacements: Record<string, string> = {
  SCRIPT: script.trimEnd(),
  SETUP: section('net = lang.Network("quickstart", dt=DT_MS * lang.ms)', 'inputs = net.input('),
  INPUTS: section('inputs = net.input(', 'cells = net.population("E", size=16, neuron=lang.COBA_LIF(tau_mem=20 * lang.ms))'),
  NETWORK: section('cells = net.population("E", size=16, neuron=lang.COBA_LIF(tau_mem=20 * lang.ms))', 'net.output("spikes", cells.spikes)'),
  OUTPUTS: section('net.output("spikes", cells.spikes)', 'bundle = lang.compile(net, target="tools/snnsim")'),
  BUNDLE: section('bundle = lang.compile(net, target="tools/snnsim")', 'result = simulate('),
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
