# RustPainter agent instructions

Codex and other coding agents must follow the repository guidance in [CLAUDE.md](CLAUDE.md). That file is the canonical shared agent guidance; this file exists so Codex discovers and applies it automatically.

## Git conventions

* After completing implementation work, create a focused git commit unless the user asks not to. Include only files relevant to that work and preserve unrelated working-tree changes.
* Never add `Co-Authored-By` or other AI/assistant attribution trailers to commits.
* Preserve the repository's existing Git author configuration. Do not modify `user.name`, `user.email`, signing configuration, or other Git identity settings.
