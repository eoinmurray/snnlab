import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';

const example = new URL('../../examples/current-lif/', import.meta.url);
const publicDir = new URL('../public/current-lif/', import.meta.url);
const script = readFileSync(new URL('current_lif.py', example), 'utf8');
const template = readFileSync(new URL('current-lif.mdx', import.meta.url), 'utf8');

function section(start: string, end: string) {
  const startIndex = script.indexOf(start);
  const endIndex = script.indexOf(end, startIndex + start.length);
  if (startIndex < 0 || endIndex < 0) throw new Error(`Missing Current LIF section: ${start}`);
  return script.slice(startIndex + start.length, endIndex).trimEnd()
    .split('\n').map(line => line.startsWith('    ') ? line.slice(4) : line).join('\n').trim();
}
const replacements: Record<string, string> = {
  SCRIPT: script.trimEnd(),
  SETUP: section('# 1. Create the network.', '# 2. Define the input'),
  INPUTS: section('# 2. Define the input and its stimulus binding.', '# 3. Define the excitatory layer'),
  NETWORK: section('# 3. Define the excitatory layer and its input projection.', '# 4. Choose outputs'),
  OUTPUTS: section('# 4. Choose outputs and exposed diagnostics.', '# 5. Compile the network'),
  BUNDLE: section('# 5. Compile the network into a bundle.', '# 6. Describe the execution'),
  EXECUTION: section('# 6. Describe the execution and simulate.', '# 7. Retrieve named results'),
  RETRIEVAL: section('# 7. Retrieve named results and select the first (only) batch item.', '# Plot the results'),
};

writeFileSync(new URL('../content/docs/current-lif.mdx', import.meta.url),
  template.replace(/\{\{(\w+)\}\}/g, (_, name) => {
    if (!(name in replacements)) throw new Error(`Unknown Current LIF placeholder: ${name}`);
    return replacements[name];
  }));
mkdirSync(publicDir, { recursive: true });
for (const name of ['current_lif.py', 'current-lif.png', 'network.png']) {
  copyFileSync(new URL(name, example), new URL(name, publicDir));
}
