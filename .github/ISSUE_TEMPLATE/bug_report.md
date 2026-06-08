---
name: Bug report
about: Report a bug in the translation system
title: ''
labels: 'bug'
assignees: ''

---

**Describe the bug**
A clear description of the bug.

**Which component?**
- [ ] Data pipeline (`uts data --pipeline`)
- [ ] Training (`uts train --full`)
- [ ] Evaluation (`uts eval --model`)
- [ ] Serving / decoder (`uts serve --decoder`)
- [ ] Coordinator (`uts serve --coordinator`)
- [ ] CLI / tools (`uts *`)
- [ ] TUI dashboard (`uts tui`)
- [ ] SDK (Android / iOS / Flutter / React Native / Web)
- [ ] Publishing (`uts publish`)
- [ ] Documentation
- [ ] CI / workflows

**To Reproduce**
Steps and the exact command run:

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened (include error output, logs).

**Environment:**
- GPU: [A100 / L4 / T4 / CPU / none]
- OS: [Linux (distro) / macOS / Windows]
- Python version:
- Command run (with flags):

**Checkpoint / resume info (if applicable):**
- Was this a fresh run or a resume?
- Did you use `--force`?
- Is there a `.pipeline_state.json` or `.checkpoints/` directory?

**Additional context**
Log files, config snippets, or screenshots (for TUI issues).
