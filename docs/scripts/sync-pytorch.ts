import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';

const example = new URL('../../examples/pytorch/', import.meta.url);
const publicDir = new URL('../public/pytorch/', import.meta.url);
const script = readFileSync(new URL('example.py', example), 'utf8');
const template = readFileSync(new URL('example.mdx', example), 'utf8');
function section(start: string, end: string, indent = true) {
  const startIndex = script.indexOf(start);
  const endIndex = script.indexOf(end, startIndex + start.length);
  if (startIndex < 0 || endIndex < 0) throw new Error(`Missing PyTorch section: ${start}`);
  return script.slice(startIndex, endIndex).trimEnd()
    .split('\n').map(line => indent && line.startsWith('    ') ? line.slice(4) : line).join('\n').trim();
}
const replacements: Record<string, string> = {
  SCRIPT: script.trimEnd(),
  DATASET: section('def make_dataset', '\nclass Classifier', false),
  CLASSIFIER: section('class Classifier', '\ndef make_diagram', false),
  EVALUATE: section('def evaluate', '\ndef main', false),
  BUNDLE: section('bundle_path = TRAINING_DIR / "network.bundle"', 'model = Classifier(bundle).to(DEVICE)'),
  MODEL: section('model = Classifier(bundle).to(DEVICE)', 'training_data = make_dataset('),
  LOADERS: section('training_data = make_dataset(', 'parameter_map = model.snn.parameter_map()'),
  OPTIMIZER: section('parameter_map = model.snn.parameter_map()', 'history = []'),
  TRAIN: section('history = []', 'weights_path = OUTPUT_DIR / "trained.pt"'),
  SAVE: section('weights_path = OUTPUT_DIR / "trained.pt"', 'metrics_path = OUTPUT_DIR / "metrics.json"'),
  METRICS: section('metrics_path = OUTPUT_DIR / "metrics.json"', 'grid = viz.FigureGrid('),
};
writeFileSync(new URL('../content/docs/pytorch-integration.mdx', import.meta.url),
  template.replace(/\{\{(\w+)\}\}/g, (_, name) => {
    if (!(name in replacements)) throw new Error(`Unknown PyTorch placeholder: ${name}`);
    return replacements[name];
  }));
mkdirSync(publicDir, { recursive: true });
for (const name of ['example.py', 'training.png', 'network.png']) {
  copyFileSync(new URL(name.endsWith('.py') ? name : `results/${name}`, example), new URL(name, publicDir));
}
