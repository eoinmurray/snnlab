import type { CollectionEntry, render } from 'astro:content';
import { parseFrontmatter } from '@astrojs/markdown-remark';
import { loader, type StaticSource } from 'fumadocs-core/source';
import { structure } from 'fumadocs-core/mdx-plugins';
import type { PageData } from './source';

export type ContentModule = {
  Content: Awaited<ReturnType<typeof render>>['Content'];
  getHeadings: () => Awaited<ReturnType<typeof render>>['headings'];
};

export function createDevSource(
  contents: Record<string, string>,
  modules: Record<string, () => Promise<ContentModule>>,
  baseUrl: string,
) {
  const files: StaticSource<{ pageData: PageData; metaData: CollectionEntry<'meta'>['data'] }>['files'] = [];
  for (const [name, raw] of Object.entries(contents)) {
    const relative = name.slice('/content/docs/'.length);
    if (relative.endsWith('.json')) {
      files.push({ type: 'meta', path: relative, data: JSON.parse(raw) });
      continue;
    }
    const { frontmatter, content } = parseFrontmatter(raw);
    const data = frontmatter as CollectionEntry<'docs'>['data'];
    const entry: CollectionEntry<'docs'> = {
      id: relative.replace(/\.(md|mdx)$/, ''), collection: 'docs',
      filePath: `content/docs/${relative}`, data, body: content,
    };
    files.push({ type: 'page', path: relative, data: {
      ...data, _raw: entry, structuredData: structure(content),
      _render: async () => {
        const module = await modules[name]();
        return { Content: module.Content, headings: module.getHeadings() };
      },
    } });
  }
  return loader({ source: { files }, baseUrl });
}
