import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';

const example = new URL('../../examples/customisation/', import.meta.url);
const publicDir = new URL('../public/customisation/', import.meta.url);
const script = readFileSync(new URL('customisation.py', example), 'utf8');
const callbacks = readFileSync(new URL('custom_neurons.py', example), 'utf8');
const template = readFileSync(new URL('customisation.mdx', import.meta.url), 'utf8');
function section(start: string, end: string, indent = false, source = script) {
  const a = source.indexOf(start), b = end ? source.indexOf(end, a + start.length) : source.length;
  if (a < 0 || b < 0) throw new Error(`Missing Customisation section: ${start}`);
  return source.slice(a + start.length, b).trimEnd().split('\n')
    .map(line => indent && line.startsWith('    ') ? line.slice(4) : line).join('\n').trim();
}
const replacements: Record<string, string> = {
  SCRIPT: script.trimEnd(),
  CALLBACKS: callbacks.trimEnd(),
  NEURON: section('# 1. Define a neuron using tensor state and a normal Python step function.', '# 2. Define an initialization', false, callbacks),
  INITIALIZER: section('# 2. Define an initialization distribution with ordinary PyTorch.', '', false, callbacks),
  REGISTRATION: section('# Register references to functions in the sibling importable module.', '\ndef main()'),
  NETWORK: section('# 3. Define a stimulus with custom tensors: quiet, then sustained activity.', '# 5. Compile and execute', true),
  EXECUTION: section('# 5. Compile and execute: the bundle stores names/config, not Python code.', '# 6. Plot the same stimulus', true),
};
writeFileSync(new URL('../content/docs/customisation.mdx', import.meta.url),
  template.replace(/\{\{(\w+)\}\}/g, (_, name) => {
    if (!(name in replacements)) throw new Error(`Unknown Customisation placeholder: ${name}`);
    return replacements[name];
  }));
mkdirSync(publicDir, { recursive: true });
for (const name of ['customisation.py', 'custom_neurons.py', 'network.png', 'customisation.png']) {
  copyFileSync(new URL(name, example), new URL(name, publicDir));
}
