# Changelog

Changes to the combined snnlab Python distribution are recorded here. Package
versions follow the policy in [VERSIONING.md](VERSIONING.md). Component and
serialized-schema versions identify separate compatibility contracts.

## [Unreleased]

### Added

1. Static Fumadocs documentation with KaTeX, search, GitHub Pages and Cloudflare hosting.
2. Complete source-derived API references for `lang`, `sim` and `viz`, with automated drift checks.
3. Six runnable general examples for bundles, simulation, input replay, training, checkpoint resume and retained-signal plotting.
4. A changelog, single-source package version and release preparation/check helper.

### Changed

1. The documentation opens at the site root; previous `/docs/` paths redirect there.
2. Package builds read the runtime version directly, and source archives exclude local documentation dependencies, caches and generated output.

## [0.1.0] - 2026-10-03

### Added

1. Initial combined Python distribution extracted from Pinglab, exposing `snnlab.lang`, `snnlab.sim` and `snnlab.viz`.
2. Portable graph authoring and validated bundles, graph-native and legacy execution, surrogate-gradient training, retained artifacts and visualization utilities.
3. Portable tests and package checks, retaining existing numerical defaults, schemas and backend identifiers.

The historical baseline is Git tag `v0.1.0`. Changes above it remain unreleased
until a subsequent version is prepared and tagged.
