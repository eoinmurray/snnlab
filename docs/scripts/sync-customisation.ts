import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';

const example = new URL('../../examples/customisation/', import.meta.url);
const publicDir = new URL('../public/customisation/', import.meta.url);
const script = readFileSync(new URL('example.py', example), 'utf8');
const callbacks = script.slice(script.indexOf('def validate_adaptive(config):'), script.indexOf('OUTPUT_DIR ='));
const template = readFileSync(new URL('example.mdx', example), 'utf8');
function section(start: string, end: string, indent = false, source = script) {
  const a = source.indexOf(start), b = end ? source.indexOf(end, a + start.length) : source.length;
  if (a < 0 || b < 0) throw new Error(`Missing Customisation section: ${start}`);
  return source.slice(a, b).trimEnd().split('\n')
    .map(line => indent && line.startsWith('    ') ? line.slice(4) : line).join('\n').trim();
}
const replacements: Record<string, string> = {
  SCRIPT: script.trimEnd(),
  CALLBACKS: callbacks.trimEnd(),
  NEURON: section('def validate_adaptive(config):', 'def validate_exponential(config):', false, callbacks),
  SYNAPSE: section('def validate_exponential(config):', 'def clipped_normal(', false, callbacks),
  INITIALIZER: section('def clipped_normal(', '', false, callbacks),
  REGISTRATION: section('def register_definitions():', '\ndef main()'),
  NETWORK: section('steps = round(DURATION_MS / DT_MS)', 'viz.render_diagram(', true),
  EXECUTION: section('viz.render_diagram(', 'import matplotlib.pyplot as plt', true),
};
writeFileSync(new URL('../content/docs/custom-neurons-and-synapses.mdx', import.meta.url),
  template.replace(/\{\{(\w+)\}\}/g, (_, name) => {
    if (!(name in replacements)) throw new Error(`Unknown Customisation placeholder: ${name}`);
    return replacements[name];
  }));
mkdirSync(publicDir, { recursive: true });
for (const name of ['example.py', 'network.png', 'customisation.png']) {
  copyFileSync(new URL(name.endsWith('.py') ? name : `results/${name}`, example), new URL(name, publicDir));
}
