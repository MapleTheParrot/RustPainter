# RustPainter agent instructions

Codex and other coding agents must follow the repository guidance in
[CLAUDE.md](CLAUDE.md). That file remains the canonical shared agent guidance;
this file exists so Codex discovers and applies it automatically.

## Git conventions

- After completing implementation work, create a focused git commit unless the
  user asks not to. Include only the files relevant to that work and preserve
  unrelated working-tree changes.
- Never add a `Co-Authored-By` trailer to commits. Commits must show only the
  repository owner as the author on GitHub. This applies to every commit,
  including ones authored entirely by an assistant.
