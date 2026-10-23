import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';

const example = new URL('../../examples/hodgkin-huxley/', import.meta.url);
const publicDir = new URL('../public/hodgkin-huxley/', import.meta.url);
const read = (name: string) => readFileSync(new URL(name, example), 'utf8');
const metrics = JSON.parse(read('comparison.json'));
const number = (value: number) => value.toPrecision(4);
const replacements: Record<string, string> = {
  SCRIPT: read('hodgkin_huxley.py').trimEnd(),
  CALLBACKS: read('hh_neuron.py').trimEnd(),
  COMPARISON: read('scipy_comparison.py').trimEnd(),
  SPIKE_COUNT: String(metrics.snnlab_spike_times_ms.length),
  SCIPY_SPIKE_COUNT: String(metrics.scipy_spike_times_ms.length),
  VOLTAGE_MAX: number(metrics.voltage_max_abs_error_mv),
  VOLTAGE_RMSE: number(metrics.voltage_rmse_mv),
  M_MAX: number(metrics.gate_max_abs_error.m),
  H_MAX: number(metrics.gate_max_abs_error.h),
  N_MAX: number(metrics.gate_max_abs_error.n),
  SPIKE_TIME_MAX: number(metrics.max_spike_time_error_ms),
};
const template = readFileSync(new URL('hodgkin-huxley.mdx', import.meta.url), 'utf8');
writeFileSync(new URL('../content/docs/hodgkin-huxley.mdx', import.meta.url),
  template.replace(/\{\{(\w+)\}\}/g, (_, name) => {
    if (!(name in replacements)) throw new Error(`Unknown HH placeholder: ${name}`);
    return replacements[name];
  }));
mkdirSync(publicDir, { recursive: true });
for (const name of [
  'hh_neuron.py', 'hodgkin_huxley.py', 'scipy_comparison.py',
  'network.png', 'hodgkin-huxley.png', 'scipy-comparison.png', 'comparison.json',
]) {
  copyFileSync(new URL(name, example), new URL(name, publicDir));
}
