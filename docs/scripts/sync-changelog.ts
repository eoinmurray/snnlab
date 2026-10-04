import { readFileSync, writeFileSync } from 'node:fs';

const source = new URL('../../CHANGELOG.md', import.meta.url);
const destination = new URL('../content/docs/changelog.mdx', import.meta.url);
const body = readFileSync(source, 'utf8')
  .replace(/^# Changelog\s*\n/, '');

writeFileSync(destination, `---
title: Changelog
description: Release history for the snnlab Python distribution.
---

${body.trim()}\n`);
