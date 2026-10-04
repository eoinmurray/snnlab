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
4. The docs npm package is tooling for the website. Its version does not identify a Python or scientific release.

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
uv run python docs/scripts/check_examples.py
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

Review the generated diff and confirm CI passes before tagging the reviewed
commit. For example, after preparing version `0.1.1`:

```sh
git add src/snnlab/__init__.py CHANGELOG.md
git commit -m "Release 0.1.1"
uv run python scripts/version.py check --tag v0.1.1
git tag -a v0.1.1 -m "snnlab 0.1.1"
git push origin main
git push origin v0.1.1
```

Replace the example version with the actual prepared version. Do not move or
reuse an existing release tag. Git tags support pinned installations without
requiring a package registry:

```sh
uv add git+https://github.com/eoinmurray/snnlab@v0.1.0
```

Publishing to PyPI or creating a GitHub Release is a separate explicit release
action; this setup does not publish automatically.
