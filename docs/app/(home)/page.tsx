import Link from 'next/link';
import { ArrowRight, Braces, Activity, ChartNoAxesCombined } from 'lucide-react';

const modules = [
  { name: 'Author', module: 'snnlab.lang', href: '/docs/authoring', icon: Braces, description: 'Build explicit circuits and compile deterministic, data-only graph bundles.' },
  { name: 'Simulate', module: 'snnlab.sim', href: '/docs/simulation', icon: Activity, description: 'Execute conductance-based networks and train with surrogate gradients.' },
  { name: 'Visualise', module: 'snnlab.viz', href: '/docs/visualisation', icon: ChartNoAxesCombined, description: 'Turn retained recordings into diagrams, figures, and animations.' },
];

export default function HomePage() {
  return (
    <main className="mx-auto w-full max-w-6xl px-6 py-20 md:py-28">
      <p className="mb-6 font-mono text-sm tracking-widest text-fd-muted-foreground uppercase">Spiking neural network laboratory</p>
      <h1 className="max-w-4xl text-5xl font-semibold tracking-tight md:text-7xl">From circuit to evidence.</h1>
      <p className="mt-8 max-w-2xl text-xl leading-relaxed text-fd-muted-foreground">A Python library for authoring, simulating, and visualising conductance-based spiking neural networks. Explicit graphs. Reproducible protocols. Inspectable results.</p>
      <div className="mt-10 flex flex-wrap items-center gap-5">
        <Link href="/docs/quickstart" className="inline-flex items-center gap-3 rounded-lg bg-fd-primary px-5 py-3 font-medium text-fd-primary-foreground">Start building <ArrowRight className="size-4" /></Link>
        <Link href="/docs" className="font-medium text-fd-muted-foreground hover:text-fd-foreground">Read the documentation</Link>
      </div>
      <div className="mt-14 overflow-x-auto rounded-xl border bg-fd-card p-5 font-mono text-sm">uv add git+https://github.com/eoinmurray/snnlab</div>
      <div className="mt-16 grid gap-5 md:grid-cols-3">
        {modules.map(({ name, module, href, icon: Icon, description }) => (
          <Link key={module} href={href} className="rounded-xl border p-7 transition-colors hover:bg-fd-muted">
            <Icon className="mb-6 size-6 text-fd-muted-foreground" />
            <h2 className="text-xl font-semibold">{name}</h2>
            <p className="mt-2 font-mono text-xs text-fd-muted-foreground">{module}</p>
            <p className="mt-4 leading-relaxed text-fd-muted-foreground">{description}</p>
          </Link>
        ))}
      </div>
      <p className="mt-10 max-w-3xl text-sm leading-relaxed text-fd-muted-foreground">Supports COBA-LIF and leaky-integrator populations, AMPA/GABA projections, and causal recurrent connections. See the documentation for execution limits and scientific contracts.</p>
    </main>
  );
}
