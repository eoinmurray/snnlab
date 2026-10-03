import type { Metadata } from 'next';
import { RootProvider } from 'fumadocs-ui/provider/next';
import 'katex/dist/katex.css';
import './global.css';

export const metadata: Metadata = {
  title: { default: 'snnlab', template: '%s | snnlab' },
  metadataBase: new URL(process.env.SITE_URL || 'https://eoinmurray.github.io/snnlab/'),
  description: 'Author, simulate, and visualise conductance-based spiking neural networks.',
};

export default function Layout({ children }: LayoutProps<'/'>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="flex flex-col min-h-screen">
        <RootProvider search={{ options: { type: 'static', api: `${process.env.NEXT_PUBLIC_BASE_PATH || ''}/api/search` } }}>{children}</RootProvider>
      </body>
    </html>
  );
}
