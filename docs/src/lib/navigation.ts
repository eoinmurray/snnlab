import type { Item } from 'fumadocs-core/page-tree';

const sidebarTitles: Record<string, string> = {
  'comparisons.mdx': 'Comparison with Other SNN Libraries',
  'snntorch-comparison.mdx': 'MNIST: SNNLab vs snnTorch',
  'shd-benchmark.mdx': 'SHD Benchmark',
};

export const sidebarLabels = {
  file(node: Item): Item {
    const title = node.$ref && sidebarTitles[node.$ref];
    return title ? { ...node, name: title } : node;
  },
};
