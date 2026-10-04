import { glob } from 'astro/loaders';
import { defineCollection } from 'astro:content';
import { z } from 'zod';

export const collections = {
  docs: defineCollection({
    loader: glob({ pattern: '**/*.{md,mdx}', base: './content/docs' }),
    schema: z.object({ title: z.string(), description: z.string().optional(), full: z.boolean().optional() }),
  }),
  meta: defineCollection({
    loader: glob({ pattern: '**/*.json', base: './content/docs' }),
    schema: z.object({
      title: z.string().optional(),
      pages: z.array(z.string()).optional(),
      collapsible: z.boolean().optional(),
      defaultOpen: z.boolean().optional(),
    }),
  }),
};
