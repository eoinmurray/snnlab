import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';

const example = new URL('../../examples/plasticity/', import.meta.url);
const publicDir = new URL('../public/plasticity/', import.meta.url);
const script = readFileSync(new URL('example.py', example), 'utf8');
const template = readFileSync(new URL('example.mdx', example), 'utf8');

function section(start: string, end: string, dedent = false) {
  const a = script.indexOf(start);
  const b = script.indexOf(end, a + start.length);
  if (a < 0 || b < 0) throw new Error(`Missing Plasticity section: ${start}`);
  return script.slice(a, b).trimEnd().split('\n')
    .map(line => dedent && line.startsWith('    ') ? line.slice(4) : line).join('\n').trim();
}

const replacements: Record<string, string> = {
  SCRIPT: script.trimEnd(),
  NETWORK: section('def make_graph(', 'def learning_rules('),
  RULES: section('def learning_rules(', 'def simulate_rule('),
  CALLBACKS: section('def initialize_hebbian(', 'def make_graph('),
  EXECUTION: section('def simulate_rule(', 'def main('),
  REGISTRATION: section('    definition = ', '    events = torch.zeros(', true),
  INPUTS: section('    events = torch.zeros(', '    fig, axes = ', true),
};

writeFileSync(new URL('../content/docs/plasticity.mdx', import.meta.url),
  template.replace(/\{\{(\w+)\}\}/g, (_, name) => {
    if (!(name in replacements)) throw new Error(`Unknown Plasticity placeholder: ${name}`);
    return replacements[name];
  }));
mkdirSync(publicDir, { recursive: true });
copyFileSync(new URL('example.py', example), new URL('example.py', publicDir));
for (const name of ['network.png', 'plasticity.png']) {
  copyFileSync(new URL(`results/${name}`, example), new URL(name, publicDir));
}
for (const stem of ['stdp', 'weight-dependent', 'triplet', 'inhibitory', 'reward-modulated', 'custom']) {
  for (const name of [`${stem}.npz`, `${stem}-rule.json`, ...(stem === 'stdp' ? [] : [`${stem}-network.png`])]) {
    copyFileSync(new URL(`results/${name}`, example), new URL(name, publicDir));
  }
}
