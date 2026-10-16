"""Generate the manual SNNLang diagram gallery; not collected by pytest."""

from __future__ import annotations

import argparse
import html
import shutil
from pathlib import Path

from cases import case_diagram, cases

from snnlab import viz

DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / "artifacts" / "visual" / "lang"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scale", type=int, default=2, help="PNG scale (default: 2)")
    args = parser.parse_args()
    if args.scale < 1:
        parser.error("--scale must be a positive integer")
    missing = [program for program in ("dot", "neato") if shutil.which(program) is None]
    if missing:
        parser.error(
            f"Graphviz executables must be available on PATH: {', '.join(missing)}"
        )

    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sections = []
    navigation = []
    for case in cases():
        for suffix in ("svg", "png"):
            (output / case.name / f"selected_groups.{suffix}").unlink(missing_ok=True)
        navigation.append(
            f'<li><a href="#{case.name}">{html.escape(case.name)}</a></li>'
        )
        figures = []
        views = (
            ("training", "circuit")
            if case.name == "all_box_types"
            else ("circuit", "training")
        )
        for view in views:
            name = view
            diagram = case_diagram(case, view)
            stem = f"{case.name}/{name}"
            for suffix in ("svg", "png"):
                viz.render_diagram(
                    diagram, output / f"{stem}.{suffix}", scale=args.scale
                )
            options = f"view={view}; all groups explicit; PNG scale={args.scale}"
            figures.append(
                f"<figure><figcaption><h3>{html.escape(name.replace('_', ' '))}</h3>"
                f"<code>{html.escape(options)}</code> "
                f'<a href="{stem}.svg">SVG</a> · <a href="{stem}.png">PNG</a>'
                f'</figcaption><a href="{stem}.svg">'
                f'<img src="{stem}.svg" alt="{html.escape(case.name)}: {name}"></a></figure>'
            )
        sections.append(
            f'<section id="{case.name}"><h2>{html.escape(case.name)}</h2>'
            f"<p>{html.escape(case.description)}</p>{''.join(figures)}</section>"
        )
    page = (
        """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SNNLang diagram review</title>
<style>
body { margin: 2rem auto; padding: 0 1rem; max-width: 1400px;
       font: 16px/1.5 system-ui, sans-serif; color: #20252b; background: #f5f6f8; }
figure { margin: 1.5rem 0; padding: 1rem; background: white;
         border: 1px solid #ccd2d9; border-radius: 8px; overflow-x: auto; }
h3 { margin: 0 0 .5rem; text-transform: capitalize; }
code { display: block; margin-bottom: .5rem; overflow-wrap: anywhere; }
img { display: block; max-width: 100%; height: auto; margin: 1rem auto 0; }
a { color: #1858a8; }
</style></head><body><h1>SNNLang diagram review</h1>
<p>Inspect labels, group boundaries, arrow routing, overlaps, and training annotations.
Open an SVG for full-size inspection; PNGs are available alongside each view.</p>
"""
        + '<nav aria-label="Cases"><ol>'
        + "".join(navigation)
        + "</ol></nav>"
        + "".join(sections)
        + "</body></html>\n"
    )
    index = output / "index.html"
    index.write_text(page, encoding="utf-8")
    print(f"Gallery: {index}")


if __name__ == "__main__":
    main()
