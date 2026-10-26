import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';

const example = new URL('../../examples/streaming/', import.meta.url);
const publicDir = new URL('../public/streaming/', import.meta.url);
const script = readFileSync(new URL('example.py', example), 'utf8');
const template = readFileSync(new URL('example.mdx', example), 'utf8');

function section(start: string, end: string) {
  const startIndex = script.indexOf(start);
  const endIndex = script.indexOf(end, startIndex + start.length);
  if (startIndex < 0 || endIndex < 0) throw new Error(`Missing Streaming section: ${start}`);
  return script.slice(startIndex, endIndex).trim();
}

const replacements: Record<string, string> = {
  SCRIPT: script.trimEnd(),
  SETTINGS: section('DT_MS =', 'def prepare_results()'),
  NETWORK: section('def make_network():', 'def incoming_windows():'),
  INPUTS: section('def incoming_windows():', 'def run_stream('),
  STREAM: section('def run_stream(', 'def plot_results('),
};

writeFileSync(new URL('../content/docs/streaming.mdx', import.meta.url),
  template.replace(/\{\{(\w+)\}\}/g, (_, name) => {
    if (!(name in replacements)) throw new Error(`Unknown Streaming placeholder: ${name}`);
    return replacements[name];
  }));
mkdirSync(publicDir, { recursive: true });
for (const name of ['example.py', 'network.png', 'streaming.png', 'streaming.npz', 'summary.json']) {
  copyFileSync(new URL(name.endsWith('.py') ? name : `results/${name}`, example), new URL(name, publicDir));
}
