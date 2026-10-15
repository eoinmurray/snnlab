import { mkdir, readdir, writeFile } from 'node:fs/promises';

// Keep bookmarks to the former directory pages working on static hosts.
export default function slashRedirects() {
  let base;
  let site;
  return {
    name: 'trailing-slash-redirects',
    hooks: {
      'astro:config:done': ({ config }) => { base = config.base; site = config.site; },
      'astro:build:done': async ({ dir }) => {
        const files = await readdir(dir, { recursive: true });
        for (const file of files) {
          if (!file.endsWith('.html') || file === '404.html' || /(^|\/)index\.html$/.test(file)) continue;
          const route = file.slice(0, -5);
          const target = `${base.replace(/\/$/, '')}/${route}`;
          const canonical = new URL(target, site).href;
          const folder = new URL(`${route}/`, dir);
          await mkdir(folder, { recursive: true });
          await writeFile(new URL('index.html', folder),
            `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=${target}"><link rel="canonical" href="${canonical}"><title>Redirecting</title></head><body><a href="${target}">Continue to the page</a></body></html>`);
        }
      },
    },
  };
}
