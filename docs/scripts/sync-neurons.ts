import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';

const example = new URL('../../examples/neurons/', import.meta.url);
const publicDir = new URL('../public/neurons/', import.meta.url);
const script = readFileSync(new URL('neurons.py', example), 'utf8');
const template = readFileSync(new URL('neurons.mdx', import.meta.url), 'utf8');

writeFileSync(new URL('../content/docs/neurons.mdx', import.meta.url),
  template.replace('{{SCRIPT}}', script.trimEnd()));
mkdirSync(publicDir, { recursive: true });
for (const name of ['neurons.py', 'neurons-cuba.png', 'network-cuba.png', 'neurons-coba.png', 'network-coba.png']) {
  copyFileSync(new URL(name, example), new URL(name, publicDir));
}
