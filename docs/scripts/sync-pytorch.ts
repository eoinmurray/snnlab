import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';

const example = new URL('../../examples/pytorch/', import.meta.url);
const publicDir = new URL('../public/pytorch/', import.meta.url);
const script = readFileSync(new URL('training.py', example), 'utf8');
const template = readFileSync(new URL('pytorch-integration.mdx', import.meta.url), 'utf8');
function section(start: string, end: string, indent = true) {
  const startIndex = script.indexOf(start);
  const endIndex = script.indexOf(end, startIndex + start.length);
  if (startIndex < 0 || endIndex < 0) throw new Error(`Missing PyTorch section: ${start}`);
  return script.slice(startIndex + start.length, endIndex).trimEnd()
    .split('\n').map(line => indent && line.startsWith('    ') ? line.slice(4) : line).join('\n').trim();
}
const replacements: Record<string, string> = {
  SCRIPT: script.trimEnd(),
  DATASET: `def make_dataset${section('def make_dataset', '\nclass Classifier', false)}`,
  CLASSIFIER: `class Classifier${section('class Classifier', '\ndef make_diagram', false)}`,
  EVALUATE: `def evaluate${section('def evaluate', '\ndef main', false)}`,
  BUNDLE: section('# 1. Load Training\'s saved bundle.', '# 2. Compose the SNN'),
  MODEL: section('# 2. Compose the SNN and ordinary PyTorch layers.', '# 3. Create sample'),
  LOADERS: section('# 3. Create sample datasets and PyTorch data loaders.', '# 4. Configure the loss'),
  OPTIMIZER: section('# 4. Configure the loss, optimizer and constraints from the recipe.', '# 5. Train with'),
  TRAIN: section('# 5. Train with an explicit PyTorch epoch and minibatch loop.', '# 6. Save a PyTorch'),
  SAVE: section('# 6. Save a PyTorch state dictionary and reload for test inference.', '# 7. Save metrics'),
  METRICS: section('# 7. Save metrics and plot the epoch curves.', '# Plot loss'),
};
writeFileSync(new URL('../content/docs/pytorch-integration.mdx', import.meta.url),
  template.replace(/\{\{(\w+)\}\}/g, (_, name) => {
    if (!(name in replacements)) throw new Error(`Unknown PyTorch placeholder: ${name}`);
    return replacements[name];
  }));
mkdirSync(publicDir, { recursive: true });
for (const name of ['training.py', 'training.png', 'network.png']) {
  copyFileSync(new URL(name, example), new URL(name, publicDir));
}
