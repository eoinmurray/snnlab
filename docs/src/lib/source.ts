import { getCollection, type CollectionEntry } from 'astro:content';
import { llms, loader, type StaticSource } from 'fumadocs-core/source';
import { structure, type StructuredData } from 'fumadocs-core/mdx-plugins';
import path from 'node:path';

type PageData = CollectionEntry<'docs'>['data'] & {
  _raw: CollectionEntry<'docs'>;
  structuredData: StructuredData;
};
const files: StaticSource<{ pageData: PageData; metaData: CollectionEntry<'meta'>['data'] }>['files'] = [];
for (const entry of await getCollection('docs')) {
  files.push({ type: 'page', path: path.relative('content/docs', entry.filePath!),
    data: { ...entry.data, _raw: entry, structuredData: structure(entry.body || '') } });
}
for (const entry of await getCollection('meta')) {
  files.push({ type: 'meta', path: path.relative('content/docs', entry.filePath!), data: entry.data });
}
export const source = loader({ source: { files }, baseUrl: import.meta.env.BASE_URL });
export const docsLlms = llms(source, {
  renderPage: (page) => `# ${page.data.title} (${page.url})\n\n${page.data._raw.body || ''}`,
});

export function markdownUrl(slugs: string[]) {
  return `${import.meta.env.BASE_URL}llms.mdx/docs/${[...slugs, 'content.md'].join('/')}`;
}
