import { copyFileSync } from 'node:fs';
import './sync-shd';

for (const [benchmark, page] of [
  ['snntorch-comparison', 'snntorch-comparison'],
  ['streaming', 'streaming-benchmark'],
]) {
  copyFileSync(
    new URL(`../../benchmarks/${benchmark}/benchmark.mdx`, import.meta.url),
    new URL(`../content/docs/${page}.mdx`, import.meta.url),
  );
}
