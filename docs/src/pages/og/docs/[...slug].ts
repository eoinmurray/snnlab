import sharp from 'sharp';
import { source } from '../../../lib/source';
import type { APIRoute } from 'astro';

export function getStaticPaths() {
  return source.getPages().map(page => ({
    params: { slug: [...page.slugs, 'image.png'].join('/') }, props: { page },
  }));
}
const escape = (text: string) => text.replace(/[<>&"]/g, c => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;' }[c]!));
export const GET: APIRoute = async ({ props }) => {
  const title = escape(props.page.data.title);
  const words = (props.page.data.description || '').split(' ');
  const lines: string[] = [''];
  for (const word of words) {
    if (lines[lines.length - 1].length + word.length > 65) lines.push('');
    lines[lines.length - 1] += `${word} `;
  }
  const description = lines.slice(0, 3).map((line, i) => `<text x="70" y="${360 + i * 40}" fill="#cccccc" font-family="sans-serif" font-size="26">${escape(line)}</text>`).join('');
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630">
    <rect width="1200" height="630" fill="#171717"/>
    <rect y="610" width="1200" height="20" fill="#c8102e"/>
    <text x="70" y="130" fill="#aaaaaa" font-family="sans-serif" font-size="32">snnlab documentation</text>
    <text x="70" y="280" fill="white" font-family="sans-serif" font-size="48">${title}</text>
    ${description}
  </svg>`;
  const image = await sharp(Buffer.from(svg)).png().toBuffer();
  return new Response(new Uint8Array(image), { headers: { 'Content-Type': 'image/png' } });
};
