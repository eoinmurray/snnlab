import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';

const example = new URL('../../examples/excalidraw/', import.meta.url);
const publicDir = new URL('../public/excalidraw/', import.meta.url);

writeFileSync(new URL('../content/docs/excalidraw.mdx', import.meta.url),
  readFileSync(new URL('example.mdx', example), 'utf8'));
mkdirSync(publicDir, { recursive: true });
for (const name of ['network.png', 'network.excalidraw', 'layout.json']) {
  copyFileSync(new URL(`results/${name}`, example), new URL(name, publicDir));
}
