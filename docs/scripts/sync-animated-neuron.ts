import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';

const example = new URL('../../examples/animated-neuron/', import.meta.url);
const publicDir = new URL('../public/animated-neuron/', import.meta.url);
const script = readFileSync(new URL('animated_neuron.py', example), 'utf8');
const template = readFileSync(new URL('animated-neuron.mdx', import.meta.url), 'utf8');

writeFileSync(new URL('../content/docs/animated-neuron.mdx', import.meta.url),
  template.replace('{{SCRIPT}}', script.trimEnd()));
mkdirSync(publicDir, { recursive: true });
for (const name of ['animated_neuron.py', 'recording.json', 'network.png']) {
  copyFileSync(new URL(name, example), new URL(name, publicDir));
}
