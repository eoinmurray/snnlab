import { syncScripts } from './content-sources.mjs';

const requested = process.argv.slice(2);
const scripts = requested.length ? [...new Set(requested)] : syncScripts;
for (const script of scripts) {
  if (!syncScripts.includes(script)) throw new Error(`Unknown documentation generator: ${script}`);
}
for (const script of scripts) {
  try {
    await import(new URL(script, import.meta.url).href);
  } catch (error) {
    throw new Error(`Failed to generate documentation with ${script}`, { cause: error });
  }
}
