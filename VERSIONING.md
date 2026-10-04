# Versioning and releases

The combined Python package uses `MAJOR.MINOR.PATCH` versions, following
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). The current version
has one source: `__version__` in `src/snnlab/__init__.py`.
[Hatchling reads that file](https://hatch.pypa.io/latest/version/) to build package
metadata; do not add a second version literal to `pyproject.toml`.

## Choosing a version

1. Patch: backward-compatible corrections, including documentation and packaging fixes.
2. Minor: new public functionality. Before 1.0, incompatible API or numerical-contract changes also require a minor bump and explicit migration notes.
3. Major: after 1.0, incompatible public API, persisted-data or numerical-contract changes. Release 1.0 when the supported public contract is ready to be stable.

Scientific compatibility includes units, timing, initialization, numerical
defaults, checkpoint authentication and retained-data interpretation. Describe
changes to these contracts explicitly in the changelog; a version bump alone
does not explain their effect or establish numerical equivalence.

## Package, component and schema identities

1. `snnlab.__version__` identifies the installable distribution and matches Git tags such as `v0.1.0`.
2. `snnlab.lang.__version__`, `snnlab.sim.__version__` and `snnlab.viz.__version__` are retained component compatibility identities. Change them deliberately when their owning contract changes; a package bump does not automatically bump them.
3. Serialized schema strings and `tools/snnsim` identify persisted scientific contracts. Change their versions only with a corresponding schema migration or explicit compatibility policy.
4. The docs JavaScript package is tooling for the website. Its version does not identify a Python or scientific release.

## Prepare a release

Use Python 3.12 for the repository tools. Add user-facing changes beneath
`## [Unreleased]` in `CHANGELOG.md` while developing. Existing released sections
should remain a historical record.

```sh
uv run python scripts/version.py check
uv run python scripts/version.py bump patch --dry-run
uv run python scripts/version.py bump patch
uv lock --check
uv run python scripts/version.py check
uv run ruff check .
uv run pytest -q -m "not slow"
(cd docs && bun run types:check && bun run build)
uv build
```

The bump helper accepts `patch`, `minor` or `major`. It updates the package
version, moves the current Unreleased notes
into a dated section, and creates an empty Unreleased section for subsequent
work. It refuses an empty release or inconsistent starting metadata. It does
not commit, tag, push or publish. The editable package record in `uv.lock` has
no version literal because its version is dynamic; version bumps do not
change dependency pins. Use `--date YYYY-MM-DD` to supply the release
date explicitly; otherwise it uses the local calendar date.

Review the generated diff before committing. For example, after preparing
version `0.1.1`:

```sh
git add src/snnlab/__init__.py CHANGELOG.md
git commit -m "Release 0.1.1"
git push origin main
```

Replace the example version with the actual prepared version. A push to `main`
that changes `src/snnlab/__init__.py` triggers `.github/workflows/publish.yml`,
following Demolab's version-triggered release pattern. It runs package checks,
builds the wheel and source archive, publishes to PyPI using Trusted Publishing,
then creates `v<VERSION>`. Forks and manual runs on other branches cannot publish.
The Actions **Run workflow** button can retry or backfill the current version;
existing PyPI files and release tags are skipped on reruns. Never move or reuse
a release tag for a different version. Git tags also support pinned installations:

```sh
uv add git+https://github.com/eoinmurray/snnlab@v0.1.0
```

## One-time publishing setup

1. Create the GitHub environment `pypi` in `eoinmurray/snnlab`.
2. In PyPI, register a Trusted Publisher for project `snnlab`: owner
   `eoinmurray`, repository `snnlab`, workflow `publish.yml`, environment `pypi`.
   If the project does not yet exist, use a pending publisher under account
   **Publishing**. If it already exists and you maintain it, use the project's
   **Publishing** settings. See the [PyPI Trusted Publishing documentation](https://docs.pypi.org/trusted-publishers/).
3. Once registered, future version changes on `main` publish automatically.
   No stored PyPI API token is required. This workflow creates a Git tag, but
   does not create a GitHub Release.
