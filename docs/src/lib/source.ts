import { getCollection, render, type CollectionEntry } from 'astro:content';
import { createDevSource, type ContentModule } from './dev-source';
import { llms, loader, type StaticSource } from 'fumadocs-core/source';
import { structure, type StructuredData } from 'fumadocs-core/mdx-plugins';
import path from 'node:path';

export type PageData = CollectionEntry<'docs'>['data'] & {
  _raw: CollectionEntry<'docs'>;
  _render?: () => Promise<Pick<Awaited<ReturnType<typeof render>>, 'Content' | 'headings'>>;
  structuredData: StructuredData;
};
// Make content and navigation files Vite dependencies so edits and added pages
// invalidate route modules during development, including their static paths.
const devContent = import.meta.env.DEV
  ? import.meta.glob<string>('/content/docs/**/*.{md,mdx,json}', {
    query: '?raw', import: 'default', eager: true,
  }) : {};
const devModules = import.meta.env.DEV
  ? import.meta.glob<ContentModule>('/content/docs/**/*.{md,mdx}') : {};

export const docsBaseUrl = `${import.meta.env.BASE_URL.replace(/\/$/, '')}/`;

export async function getSource() {
  const files: StaticSource<{ pageData: PageData; metaData: CollectionEntry<'meta'>['data'] }>['files'] = [];
  if (import.meta.env.DEV) {
    // Use Vite's current file snapshot instead of the collection's cached store.
    return createDevSource(devContent, devModules, import.meta.env.BASE_URL);
  }
  const [pages, metadata] = await Promise.all([getCollection('docs'), getCollection('meta')]);
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
