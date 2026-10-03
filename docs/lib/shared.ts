import { createGetUrl } from 'fumadocs-core/source';

export const appName = 'snnlab';
export const docsRoute = '/';
export const docsImageRoute = '/og/docs';
export const docsContentRoute = '/llms.mdx/docs';

export const gitConfig = {
  user: 'eoinmurray',
  repo: 'snnlab',
  branch: 'main',
};

const getContentUrl = createGetUrl(docsContentRoute);

export function getPageMarkdownUrl(page: { slugs: string[]; locale?: string }) {
  const segments = [...page.slugs, 'content.md'];

  return { segments, url: `${process.env.NEXT_PUBLIC_BASE_PATH || ''}${getContentUrl(segments, page.locale)}` };
}

const getImageUrl = createGetUrl(docsImageRoute);

export function getPageImageUrl(page: { slugs: string[]; locale?: string }) {
  const segments = [...page.slugs, 'image.png'];

  return { segments, url: `${process.env.NEXT_PUBLIC_BASE_PATH || ''}${getImageUrl(segments, page.locale)}` };
}
