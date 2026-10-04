import { getDocsLlms } from '../lib/source';
export const GET = async () => new Response(await (await getDocsLlms()).index(), { headers: { 'Content-Type': 'text/plain' } });
