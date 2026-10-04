import { docsLlms } from '../lib/source';
export const GET = async () => new Response(await docsLlms.index(), { headers: { 'Content-Type': 'text/plain' } });
