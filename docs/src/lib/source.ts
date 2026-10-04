import { getCollection, type CollectionEntry } from 'astro:content';
import { llms, loader, type StaticSource } from 'fumadocs-core/source';
import { structure, type StructuredData } from 'fumadocs-core/mdx-plugins';
import path from 'node:path';

type PageData = CollectionEntry<'docs'>['data'] & {
  _raw: CollectionEntry<'docs'>;
  structuredData: StructuredData;
};
// Make content and navigation files Vite dependencies so edits and added pages
// invalidate route modules during development, including their static paths.
if (import.meta.env.DEV) {
  import.meta.glob('/content/docs/**/*.{md,mdx,json}', {
    query: '?raw', import: 'default', eager: true,
  });
}

export const docsBaseUrl = `${import.meta.env.BASE_URL.replace(/\/$/, '')}/`;

export async function getSource() {
  const [pages, metadata] = await Promise.all([getCollection('docs'), getCollection('meta')]);
  const files: StaticSource<{ pageData: PageData; metaData: CollectionEntry<'meta'>['data'] }>['files'] = [];
  for (const entry of pages) {
    files.push({ type: 'page', path: path.relative('content/docs', entry.filePath!),
      data: { ...entry.data, _raw: entry, structuredData: structure(entry.body || '') } });
  }
  for (const entry of metadata) {
    files.push({ type: 'meta', path: path.relative('content/docs', entry.filePath!), data: entry.data });
  }
  return loader({ source: { files }, baseUrl: import.meta.env.BASE_URL });
}

export async function getDocsLlms() {
  return llms(await getSource(), {
    renderPage: (page) => `# ${page.data.title} (${page.url})\n\n${page.data._raw.body || ''}`,
  });
}

export function markdownUrl(slugs: string[]) {
  return `${docsBaseUrl}llms.mdx/docs/${[...slugs, 'content.md'].join('/')}`;
}
