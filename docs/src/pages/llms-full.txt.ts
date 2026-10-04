import { getDocsLlms } from '../lib/source';
export const GET = async () => new Response(await (await getDocsLlms()).full(), { headers: { 'Content-Type': 'text/plain' } });
