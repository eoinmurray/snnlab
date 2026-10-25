import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';

const example = new URL('../../examples/meanfield/', import.meta.url);
const publicDir = new URL('../public/meanfield/', import.meta.url);
const script = readFileSync(new URL('example.py', example), 'utf8');
const template = readFileSync(new URL('example.mdx', example), 'utf8');
type Equation = { lhs_latex: string; rhs_latex: string; latex: string };
const model = JSON.parse(readFileSync(new URL('results/model.json', example), 'utf8')) as {
  schema: string;
  state_order: string[];
  states: Record<string, { latex: string }>;
  equations: Record<string, Equation>;
  symbols: Record<string, { definition_latex?: string }>;
};
if (model.schema !== 'snnlab.meanfield.model/v1') throw new Error('Unsupported mean-field model document');
const couplingDefinition = model.symbols['E_to_I.coupling']?.definition_latex;
if (!couplingDefinition) throw new Error('Missing mean incoming coupling definition');

function equationBlock(ids: string[]) {
  const rows = ids.map(id => {
    const equation = model.equations[id];
    if (!equation) throw new Error(`Missing mean-field equation: ${id}`);
    return `${equation.lhs_latex} &= ${equation.rhs_latex}`;
  });
  return '\\begin{aligned}\n' + rows.join(' \\\\\n') + '\n\\end{aligned}';
}

function section(start: string, end: string) {
  const startIndex = script.indexOf(start);
  const endIndex = script.indexOf(end, startIndex + start.length);
  if (startIndex < 0 || endIndex < 0) throw new Error(`Missing mean-field section: ${start}`);
  return script.slice(startIndex, endIndex).trimEnd();
}

const replacements: Record<string, string> = {
  MODEL_STATES: 'x=\\left(' + model.state_order.map(id => model.states[id].latex).join(',') + '\\right)^{\\mathsf T}',
  MODEL_COUPLING_DEFINITION: couplingDefinition,
  MODEL_RATES: equationBlock(['E.rate', 'I.rate']),
  MODEL_CONDUCTANCES: equationBlock(['E_to_E.conductance', 'E_to_I.conductance', 'I_to_E.conductance', 'I_to_I.conductance']),
  MODEL_CURRENTS: equationBlock(['E.effective_current', 'I.effective_current']),
  MODEL_EXTERNAL_INPUT: equationBlock(['X_to_E.mean_conductance', 'X_to_I.mean_conductance']),
  MODEL_DRIVING_FORCES: equationBlock(['E.driving_force_exc', 'E.driving_force_inh', 'I.driving_force_exc', 'I.driving_force_inh']),
  MODEL_BUILDING: section('    model = mf.reduce(', '    control = mf.PiecewiseLinear(')
    .split('\n').map(line => line.startsWith('    ') ? line.slice(4) : line).join('\n'),
};

writeFileSync(new URL('../content/docs/meanfield.mdx', import.meta.url),
  template.replace(/\{\{(\w+)\}\}/g, (_, name) => {
    if (!(name in replacements)) throw new Error(`Unknown mean-field placeholder: ${name}`);
    return replacements[name];
  }));
mkdirSync(publicDir, { recursive: true });
for (const name of ['example.py', 'network.png', 'analysis.png', 'analysis.json', 'model.json']) {
  copyFileSync(new URL(name.endsWith('.py') ? name : `results/${name}`, example), new URL(name, publicDir));
}
