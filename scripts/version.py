"""Check package release metadata or prepare a major/minor/patch release."""

from __future__ import annotations

import argparse
import ast
import re
from datetime import date
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
VERSION = re.compile(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)")


def package_version(root: Path) -> str:
    tree = ast.parse((root / "src/snnlab/__init__.py").read_text())
    values = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets)
    ]
    if (
        len(values) != 1
        or not isinstance(values[0], str)
        or not VERSION.fullmatch(values[0])
    ):
        raise ValueError("Expected one MAJOR.MINOR.PATCH __version__ assignment")
    return values[0]


def check(root: Path, tag: str | None = None) -> str:
    version = package_version(root)
    project = tomllib.loads((root / "pyproject.toml").read_text())
    if "version" in project["project"] or "version" not in project["project"].get(
        "dynamic", []
    ):
        raise ValueError(
            "pyproject.toml must declare a dynamic version, without a second literal"
        )
    if project["tool"]["hatch"]["version"]["path"] != "src/snnlab/__init__.py":
        raise ValueError("Hatchling must read the package's version source")
    lock = tomllib.loads((root / "uv.lock").read_text())
    records = [p for p in lock["package"] if p["name"] == "snnlab"]
    if (
        len(records) != 1
        or records[0].get("source") != {"editable": "."}
        or "version" in records[0]
    ):
        raise ValueError(
            "uv.lock must retain a dynamic editable snnlab record without a second version"
        )
    changelog = (root / "CHANGELOG.md").read_text()
    if changelog.count("## [Unreleased]\n") != 1:
        raise ValueError("Changelog must have exactly one Unreleased section")
    headings = re.findall(r"^## \[([^\]]+)\] - (\d{4}-\d{2}-\d{2})$", changelog, re.M)
    if not headings or headings[0][0] != version:
        raise ValueError(
            "Newest dated changelog section must match the package version"
        )
    if len({v for v, _ in headings}) != len(headings):
        raise ValueError("Duplicate release version in changelog")
    for released, day in headings:
        if not VERSION.fullmatch(released):
            raise ValueError(f"Invalid changelog version: {released}")
        date.fromisoformat(day)
    if tag is not None and tag != f"v{version}":
        raise ValueError(f"Expected tag v{version}, got {tag}")
    return version


def prepare(root: Path, part: str, day: date) -> tuple[str, dict[Path, str]]:
    current = check(root)
    segments = [int(x) for x in current.split(".")]
    index = {"major": 0, "minor": 1, "patch": 2}[part]
    segments[index] += 1
    segments[index + 1 :] = [0] * (2 - index)
    target = ".".join(map(str, segments))
    path = root / "CHANGELOG.md"
    changelog = path.read_text()
    start = changelog.index("## [Unreleased]\n") + len("## [Unreleased]\n")
    end = changelog.index("\n## [", start)
    notes = changelog[start:end].strip()
    if not any(
        line.strip() and not line.startswith("#") for line in notes.splitlines()
    ):
        raise ValueError("Refusing to prepare a release with no Unreleased notes")
    previous_date = re.search(r"^## \[[^\]]+\] - (\d{4}-\d{2}-\d{2})$", changelog, re.M)
    if day < date.fromisoformat(previous_date[1]):
        raise ValueError("Release date cannot precede the latest release")
    changes = {
        path: changelog[:start]
        + f"\n## [{target}] - {day.isoformat()}\n\n{notes}\n"
        + changelog[end:]
    }
    path = root / "src/snnlab/__init__.py"
    text, count = re.subn(
        r"^__version__\s*=\s*[\'\"][^\'\"]+[\'\"]",
        f'__version__ = "{target}"',
        path.read_text(),
        flags=re.M,
    )
    if count != 1:
        raise ValueError("Could not update exactly one version assignment")
    changes[path] = text
    return target, changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("check")
    verify.add_argument("--tag")
    bump = commands.add_parser("bump")
    bump.add_argument("part", choices=("major", "minor", "patch"))
    bump.add_argument("--date", type=date.fromisoformat, default=date.today())
    bump.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "check":
            print(f"Version metadata consistent: {check(ROOT, args.tag)}")
        else:
            version, changes = prepare(ROOT, args.part, args.date)
            for path, content in changes.items():
                if not args.dry_run:
                    path.write_text(content)
                print(
                    f"{'Would update' if args.dry_run else 'Updated'} {path.relative_to(ROOT)}"
                )
            print(
                f"{'Would prepare' if args.dry_run else 'Prepared'} {version}; review before tagging"
            )
    except (ValueError, KeyError) as exc:
        parser.exit(1, f"Version check failed: {exc}\n")


if __name__ == "__main__":
    main()
