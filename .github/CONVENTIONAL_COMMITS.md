# Conventional Commits Guide

Use these prefixes in commit messages to enable automatic versioning and changelog generation:

- feat: a new feature (minor bump)
- fix: a bug fix (patch bump)
- perf: a performance improvement (patch bump)
- refactor: code refactoring (no public API change)
- docs: documentation only changes
- test: adding or fixing tests
- chore: tooling, build, CI, or maintenance
- BREAKING CHANGE: include in body or use ! after type (e.g., feat!:) for major bump

Examples:
- feat(web): add streaming translation API
- fix(python): handle empty vocab manifest gracefully
- chore(ci): add npm publish workflow
- feat!: switch encoder to new embedding format

Formatting:
- type(scope): summary
- Blank line
- Optional detailed description and BREAKING CHANGE paragraph