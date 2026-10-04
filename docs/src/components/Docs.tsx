import { RootProvider } from 'fumadocs-ui/provider/astro';
import { DocsLayout } from 'fumadocs-ui/layouts/docs';
import { DocsPage, DocsTitle, DocsDescription, MarkdownCopyButton, type DocsPageProps } from 'fumadocs-ui/layouts/docs/page';
import { ViewOptionsPopover } from './ViewOptionsPopover';
import type { Root } from 'fumadocs-core/page-tree';
import type { ReactNode } from 'react';
import { navigate } from 'astro:transitions/client';

export function Docs({ tree, pathname, page, markdown, file, title, description, children }: {
  tree: Root; pathname: string; page: DocsPageProps; markdown: string; file: string; title: string; description?: string; children: ReactNode;
}) {
  return <RootProvider pathname={pathname} navigate={navigate}
    search={{ options: { type: 'static', api: `${import.meta.env.BASE_URL}api/search` } }}>
    <DocsLayout tree={tree} nav={{ title: 'snnlab', url: import.meta.env.BASE_URL }} githubUrl="https://github.com/eoinmurray/snnlab">
      <DocsPage {...page}>
        <DocsTitle>{title}</DocsTitle>
        <DocsDescription className="mb-0">{description}</DocsDescription>
        <div className="flex flex-row gap-2 items-center border-b pb-6">
          <MarkdownCopyButton markdownUrl={markdown} />
          <ViewOptionsPopover pathname={pathname} markdownUrl={markdown} githubUrl={`https://github.com/eoinmurray/snnlab/blob/main/docs/content/docs/${file}`} />
        </div>
        {children}
      </DocsPage>
    </DocsLayout>
  </RootProvider>;
}
