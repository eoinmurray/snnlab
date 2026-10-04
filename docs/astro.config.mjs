import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import mdx from '@astrojs/mdx';
import tailwindcss from '@tailwindcss/vite';
import { unified } from '@astrojs/markdown-remark';
import { rehypeCode, remarkGfm, remarkHeading, remarkStructure } from 'fumadocs-core/mdx-plugins';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const base = process.env.SITE_BASE_PATH || '/';

function baseLinks() {
  return (tree) => {
    function visit(node) {
      if (node.properties) {
        for (const key of ['href', 'src']) {
          const value = node.properties[key];
          if (typeof value === 'string' && value.startsWith('/') && !value.startsWith('//')) {
            node.properties[key] = `${base.replace(/\/$/, '')}${value}`;
          }
        }
      }
      node.children?.forEach(visit);
    }
    visit(tree);
  };
}

export default defineConfig({
  output: 'static',
  outDir: './out',
  site: process.env.SITE_URL || 'https://ssnlab.eoinmurray.info',
  base,
  trailingSlash: 'always',
  markdown: {
    processor: unified({
      syntaxHighlight: false,
      remarkPlugins: [remarkGfm, remarkMath, remarkHeading, [remarkStructure, { exportAs: 'structuredData' }]],
      rehypePlugins: [[rehypeKatex, { strict: 'error', throwOnError: true }], rehypeCode, baseLinks],
    }),
  },
  integrations: [react(), mdx({ extendMarkdownConfig: true, syntaxHighlight: false })],
  vite: { plugins: [tailwindcss()] },
});
