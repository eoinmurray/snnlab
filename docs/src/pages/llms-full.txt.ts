import { docsLlms } from '../lib/source';
export const GET = async () => new Response(await docsLlms.full(), { headers: { 'Content-Type': 'text/plain' } });
