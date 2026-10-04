import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';

const example = new URL('../../examples/training/', import.meta.url);
const publicDir = new URL('../public/training/', import.meta.url);
const script = readFileSync(new URL('training.py', example), 'utf8');
const template = readFileSync(new URL('training.mdx', import.meta.url), 'utf8');

function section(start: string, end: string, indent = true) {
  const startIndex = script.indexOf(start);
  const endIndex = script.indexOf(end, startIndex + start.length);
  if (startIndex < 0 || endIndex < 0) throw new Error(`Missing Training section: ${start}`);
  return script.slice(startIndex + start.length, endIndex).trimEnd()
    .split('\n').map(line => indent && line.startsWith('    ') ? line.slice(4) : line).join('\n').trim();
}
const replacements: Record<string, string> = {
  SCRIPT: script.trimEnd(),
  DATASET: `def make_dataset${section('def make_dataset', '\ndef main', false)}`,
  SETUP: section('# 1. Create the network.', '# 2. Define inputs'),
  INPUTS: section('# 2. Define inputs, labelled datasets and bindings.', '# 3. Define the excitatory'),
  NETWORK: section('# 3. Define the excitatory layer and two-class readout.', '# 4. Declare the official'),
  OUTPUTS: section('# 4. Declare the official output and optional diagnostics.', '# 5. Define the loss'),
  RECIPE: section('# 5. Define the loss, trainable parameters and optimizer.', '# 6. Compile and save'),
  BUNDLE: section('# 6. Compile and save the bundle for training and later inference.', '# 7. Train and measure'),
  TRAIN: section('# 7. Train and measure each epoch in one call.', '# 8. Save epoch'),
  METRICS: section('# 8. Save epoch metrics and plot training curves.', '# Plot the epoch'),
};
writeFileSync(new URL('../content/docs/training.mdx', import.meta.url),
  template.replace(/\{\{(\w+)\}\}/g, (_, name) => {
    if (!(name in replacements)) throw new Error(`Unknown Training placeholder: ${name}`);
    return replacements[name];
  }));
mkdirSync(publicDir, { recursive: true });
for (const name of ['training.py', 'training.png', 'network.png']) {
  copyFileSync(new URL(name, example), new URL(name, publicDir));
}
