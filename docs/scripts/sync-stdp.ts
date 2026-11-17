import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';

const example = new URL('../../examples/stdp/', import.meta.url);
const publicDir = new URL('../public/stdp/', import.meta.url);
const script = readFileSync(new URL('example.py', example), 'utf8');
const template = readFileSync(new URL('example.mdx', example), 'utf8');
const summary = JSON.parse(readFileSync(new URL('results/summary.json', example), 'utf8'));

function section(start: string, end: string, dedent = true) {
  const startIndex = script.indexOf(start);
  const endIndex = script.indexOf(end, startIndex + start.length);
  if (startIndex < 0 || endIndex < 0) throw new Error(`Missing STDP section: ${start}`);
  const text = script.slice(startIndex, endIndex).trimEnd();
  if (!dedent) return text.trim();
  return text.split('\n').map(line => line.startsWith('    ') ? line.slice(4) : line).join('\n').trim();
}

type Experiment = {
  label: string;
  output_rate_before_hz: number;
  output_rate_after_hz: number;
  mean_weight_first_half: number;
  mean_weight_second_half: number;
  'fraction_below_0.1': number;
  'fraction_above_0.9': number;
};
const fixed = (value: number, digits: number) => value.toFixed(digits);
const percent = (value: number) => `${Math.round(100 * value)}%`;
const table = [
  '| Experiment | Output rate before (Hz) | Output rate after (Hz) | Mean weight, inputs 0–499 | Mean weight, inputs 500–999 | Below 0.1 | Above 0.9 |',
  '| --- | --- | --- | --- | --- | --- | --- |',
  ...summary.experiments.map((e: Experiment) => [
    e.label,
    fixed(e.output_rate_before_hz, 1),
    fixed(e.output_rate_after_hz, 1),
    fixed(e.mean_weight_first_half, 2),
    fixed(e.mean_weight_second_half, 2),
    percent(e['fraction_below_0.1']),
    percent(e['fraction_above_0.9']),
  ].join(' | ')).map((row: string) => `| ${row} |`),
].join('\n');

const replacements: Record<string, string> = {
  SCRIPT: script.trimEnd(),
  CONSTANTS: section('DT_MS = 0.5', 'GROUP = N_EXC // 2', false),
  INPUTS: section('exc = net.input(', 'cell = net.population('),
  NETWORK: section('cell = net.population(', 'net.output('),
  INPUT_CHUNK: section('def input_chunk(', 'def learn(', false),
  BUILD: section('bundle = lang.compile(', 'initial = model.parameter_map()'),
  LEARN: section('def learn(', 'def probe(', false),
  PROBE: section('def probe(', 'def learning_window(', false),
  RUN: section('rng = np.random.default_rng(SEED)', 'return {'),
  WINDOW: section('def learning_window(', 'def run_example(', false),
  LEARNING_SECONDS: String(summary.learning_seconds),
  SUMMARY_TABLE: table,
};

writeFileSync(new URL('../content/docs/stdp.mdx', import.meta.url),
  template.replace(/\{\{(\w+)\}\}/g, (_, name) => {
    if (!(name in replacements)) throw new Error(`Unknown STDP placeholder: ${name}`);
    return replacements[name];
  }));
mkdirSync(publicDir, { recursive: true });
for (const name of ['example.py', 'network.png', 'stdp-window.png', 'stdp-learning.png', 'summary.json']) {
  copyFileSync(new URL(name.endsWith('.py') ? name : `results/${name}`, example), new URL(name, publicDir));
}
