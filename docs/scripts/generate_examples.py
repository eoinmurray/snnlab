"""Keep example pages and Python downloads synchronized with runnable scripts."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTES = json.loads(Path(__file__).with_name("example_notes.json").read_text())
DEST = ROOT / "docs/content/docs/examples"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = {}
    for note in NOTES:
        name = note["name"]
        script = (ROOT / "examples" / f"{name}.py").read_text()
        page = (
            f"---\ntitle: {note['title']}\ndescription: {note['description']}\n---\n\n"
        )
        page += note["intro"] + "\n\n"
        page += f"[Download the standalone script](/examples/{name}.py) · [View source](https://github.com/eoinmurray/snnlab/blob/main/examples/{name}.py)\n\n"
        page += "## Run it\n\nAfter [installation](/installation), run from the repository root:\n\n"
        page += f"```sh\nuv run python examples/{name}.py\n```\n\n"
        page += "For the download, save the Python file and run it in an environment with snnlab installed. Every example is self-contained, uses CPU, and runs without a dataset download, Graphviz or FFmpeg.\n\n"
        page += f"Files, when produced, are written below `artifacts/examples/{name}/` relative to your working directory. A repeated run replaces that example’s outputs.\n\n"
        page += f"## Complete code\n\n```python\n{script.rstrip()}\n```\n\n"
        page += note["details"] + "\n\n## Expected result\n\n"
        page += f"```text\n{note['output']}\n```\n\n"
        if note.get("preview"):
            page += "![Spike raster for six simulated cells above their mean membrane voltage over a 100 millisecond presentation.](/examples/recording.png)\n\n"
        page += "## API links\n\n" + note["api"] + ".\n"
        expected[DEST / f"{name}.mdx"] = page
        expected[ROOT / "docs/public/examples" / f"{name}.py"] = script
    expected[DEST / "meta.json"] = (
        json.dumps(
            {"title": "Examples", "pages": ["index"] + [n["name"] for n in NOTES]},
            indent=2,
        )
        + "\n"
    )
    overview = """---
title: Examples
description: Small runnable examples for authoring, simulation, training and visualisation.
---

These standalone examples demonstrate the general workflow with small CPU runs. Each page contains complete code, a Python download, expected results and links to its API contracts. No dataset download is required.

## Choose an example

| Example | Learn how to | Produces |
| --- | --- | --- |
| [Build a portable circuit](/examples/build_bundle) | Author, compile, reload and plan a graph | A portable bundle |
| [Simulate a driven E/I circuit](/examples/poisson_circuit) | Generate Poisson input and inspect spikes | An activity archive |
| [Replay dense and sparse spikes](/examples/replay_inputs) | Bind an exact event stream in two representations | Checked raw counts |
| [Train a two-class readout](/examples/train_readout) | Declare a recipe, update weights and infer | A training checkpoint |
| [Save and resume training](/examples/resume_training) | Continue training and compare to a full run | A checkpoint and equality checks |
| [Plot retained activity](/examples/plot_recording) | Turn retained arrays into a raster and voltage figure | A PNG and recording archive |

## Getting started

1. [Install snnlab](/installation), or run `uv sync` from a repository checkout.
2. Choose a page and run its script from the repository root. Downloads also run in an environment with snnlab installed.
3. Inspect the printed results and any files under `artifacts/examples/`.
4. Change one declared setting, such as the input rate, timestep or learning rate, and rerun the example.

The examples use explicit graph execution, physical timing and small synthetic fixtures. They teach interfaces; they do not report dataset accuracy, establish scientific acceptance criteria or guarantee cross-device equality. The training example evaluates its own fixture and says so explicitly.

## Check every example

```sh
uv run python docs/scripts/check_examples.py
```

This runs the exact downloadable scripts in a temporary directory and checks their assertions. The package CI runs the same command. The rendered pages and downloads are generated from the scripts so copying code from a page cannot drift away from its executable source.
"""
    expected[DEST / "index.mdx"] = overview
    stale = []
    for path, content in expected.items():
        if args.check:
            if not path.exists() or path.read_text() != content:
                stale.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    if stale:
        raise SystemExit("Stale examples:\n" + "\n".join(stale))
    print(
        f"{'Checked' if args.check else 'Generated'} {len(NOTES)} example pages and downloads"
    )


if __name__ == "__main__":
    main()
