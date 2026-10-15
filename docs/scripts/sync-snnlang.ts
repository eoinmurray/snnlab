import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';

const example = new URL('../../examples/snnlang/', import.meta.url);
const publicDir = new URL('../public/snnlang/', import.meta.url);
mkdirSync(publicDir, { recursive: true });
let page = readFileSync(new URL('snnlang.mdx', import.meta.url), 'utf8');

for (const name of ['feedforward', 'recurrent']) {
  const script = readFileSync(new URL(`${name}.py`, example), 'utf8');
  const placeholder = `{{${name.toUpperCase()}}}`;
  if (!page.includes(placeholder)) throw new Error(`Missing SNNLang script placeholder: ${name}`);
  page = page.replace(placeholder, () => script.trimEnd());
  for (const extension of ['py', 'png']) {
    copyFileSync(new URL(`${name}.${extension}`, example), new URL(`${name}.${extension}`, publicDir));
  }
}
writeFileSync(new URL('../content/docs/snnlang.mdx', import.meta.url), page);
