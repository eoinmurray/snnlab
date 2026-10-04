import { getSource, getDocsLlms } from '../../../lib/source';
import type { APIRoute } from 'astro';

export async function getStaticPaths() {
  const source = await getSource();
  return source.getPages().map(page => ({
    params: { slug: [...page.slugs, 'content.md'].join('/') }, props: { page },
  }));
}
export const GET: APIRoute = async ({ props }) => new Response(await (await getDocsLlms()).page(props.page), {
  headers: { 'Content-Type': 'text/markdown' },
});
