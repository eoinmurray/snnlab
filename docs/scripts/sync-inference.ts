import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';

const example = new URL('../../examples/inference/', import.meta.url);
const publicDir = new URL('../public/inference/', import.meta.url);
const script = readFileSync(new URL('inference.py', example), 'utf8');
const template = readFileSync(new URL('inference.mdx', import.meta.url), 'utf8');

function section(start: string, end: string, indent = true) {
  const startIndex = script.indexOf(start);
  const endIndex = script.indexOf(end, startIndex + start.length);
  if (startIndex < 0 || endIndex < 0) throw new Error(`Missing Inference section: ${start}`);
  return script.slice(startIndex + start.length, endIndex).trimEnd()
    .split('\n').map(line => indent && line.startsWith('    ') ? line.slice(4) : line).join('\n').trim();
}
const replacements: Record<string, string> = {
  SCRIPT: script.trimEnd(),
  DATASET: `def make_dataset${section('def make_dataset', '\ndef main', false)}`,
  ARTIFACTS: section('# 1. Locate the saved training artifacts.', '# 2. Load the saved'),
  NETWORK: section('# 2. Load the saved network and generate its diagram.', '# 3. Generate fresh'),
  INPUTS: section('# 3. Generate fresh inputs and bind them to the saved network.', '# 4. Load learned'),
  INFER: section('# 4. Load learned weights and run inference.', '# 5. Retrieve scores'),
  RESULTS: section('# 5. Retrieve scores, choose classes and measure accuracy.', '# 6. Save predictions'),
  SAVE: section('# 6. Save predictions and checkpoint provenance.', '# Plot one fresh'),
};
writeFileSync(new URL('../content/docs/inference.mdx', import.meta.url),
  template.replace(/\{\{(\w+)\}\}/g, (_, name) => {
    if (!(name in replacements)) throw new Error(`Unknown Inference placeholder: ${name}`);
    return replacements[name];
  }));
mkdirSync(publicDir, { recursive: true });
for (const name of ['inference.py', 'inference.png', 'network.png']) {
  copyFileSync(new URL(name, example), new URL(name, publicDir));
}
