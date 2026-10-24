import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';

const example = new URL('../../examples/inference/', import.meta.url);
const publicDir = new URL('../public/inference/', import.meta.url);
const script = readFileSync(new URL('example.py', example), 'utf8');
const template = readFileSync(new URL('inference.mdx', import.meta.url), 'utf8');

function section(start: string, end: string, indent = true) {
  const startIndex = script.indexOf(start);
  const endIndex = script.indexOf(end, startIndex + start.length);
  if (startIndex < 0 || endIndex < 0) throw new Error(`Missing Inference section: ${start}`);
  return script.slice(startIndex, endIndex).trimEnd()
    .split('\n').map(line => indent && line.startsWith('    ') ? line.slice(4) : line).join('\n').trim();
}
const replacements: Record<string, string> = {
  SCRIPT: script.trimEnd(),
  DATASET: section('def make_dataset', '\ndef main', false),
  ARTIFACTS: section('bundle_path = TRAINING_DIR / "network.bundle"', 'bundle = lang.load_bundle(bundle_path)'),
  NETWORK: section('bundle = lang.load_bundle(bundle_path)', 'input_spikes, labels = make_dataset(TEST_SAMPLES, seed=DATA_SEED)'),
  INPUTS: section('input_spikes, labels = make_dataset(TEST_SAMPLES, seed=DATA_SEED)', 'with torch.no_grad():'),
  SIMULATE: section('with torch.no_grad():', 'data = result.numpy()'),
  RESULTS: section('data = result.numpy()', 'predictions_path = OUTPUT_DIR / "predictions.json"'),
  SAVE: section('predictions_path = OUTPUT_DIR / "predictions.json"', 'grid = viz.FigureGrid('),
};
writeFileSync(new URL('../content/docs/inference.mdx', import.meta.url),
  template.replace(/\{\{(\w+)\}\}/g, (_, name) => {
    if (!(name in replacements)) throw new Error(`Unknown Inference placeholder: ${name}`);
    return replacements[name];
  }));
mkdirSync(publicDir, { recursive: true });
for (const name of ['example.py', 'inference.png', 'network.png']) {
  copyFileSync(new URL(name.endsWith('.py') ? name : `results/${name}`, example), new URL(name, publicDir));
}
