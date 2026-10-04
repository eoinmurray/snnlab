import { source, docsLlms } from '../../../lib/source';
import type { APIRoute } from 'astro';

export function getStaticPaths() {
  return source.getPages().map(page => ({
    params: { slug: [...page.slugs, 'content.md'].join('/') }, props: { page },
  }));
}
export const GET: APIRoute = async ({ props }) => new Response(await docsLlms.page(props.page), {
  headers: { 'Content-Type': 'text/markdown' },
});
