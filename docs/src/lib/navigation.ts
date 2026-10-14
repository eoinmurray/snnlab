import type { Item } from 'fumadocs-core/page-tree';

const sidebarTitles: Record<string, string> = {
  'data-formats.mdx': 'Output data',
  'comparisons.mdx': 'Library comparison',
  'snntorch-comparison.mdx': 'MNIST with snnTorch',
};

export const sidebarLabels = {
  file(node: Item): Item {
    const title = node.$ref && sidebarTitles[node.$ref];
    return title ? { ...node, name: title } : node;
  },
};
