# Changelog

Changes to the combined snnlab Python distribution are recorded here. Package
versions follow the policy in [VERSIONING.md](VERSIONING.md). Component and
serialized-schema versions identify separate compatibility contracts.

## [Unreleased]

### Changed

1. Removed Scira AI and Cursor from the documentation's Open menu.

## [0.1.1] - 2026-10-04

### Added

1. Static Fumadocs documentation with KaTeX, search, GitHub Pages and Cloudflare hosting.
2. Complete source-derived API references for `lang`, `sim` and `viz`, with automated drift checks.
3. Six runnable general examples for bundles, simulation, input replay, training, checkpoint resume and retained-signal plotting.
4. A changelog, single-source package version and release preparation/check helper.
5. Automatic PyPI publishing on version-source changes to `main`, using Trusted Publishing and release tags after package checks pass.

### Changed

1. The documentation opens at the site root; previous `/docs/` paths redirect there.
2. Documentation builds use Astro static output with Fumadocs React islands in place of Next.js.
3. Package builds read the runtime version directly, and source archives exclude local documentation dependencies, caches and generated output.

## [0.1.0] - 2026-10-03

### Added

1. Initial combined Python distribution extracted from Pinglab, exposing `snnlab.lang`, `snnlab.sim` and `snnlab.viz`.
2. Portable graph authoring and validated bundles, graph-native and legacy execution, surrogate-gradient training, retained artifacts and visualization utilities.
3. Portable tests and package checks, retaining existing numerical defaults, schemas and backend identifiers.

The historical baseline is Git tag `v0.1.0`. Changes above it remain unreleased
until a subsequent version is prepared and tagged.
