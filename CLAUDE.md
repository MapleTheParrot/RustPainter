# RustPainter

## Git conventions

* Never add `Co-Authored-By` or other AI/assistant attribution trailers to commits.
* Preserve the repository's existing Git author configuration. Do not modify `user.name`, `user.email`, signing configuration, or other Git identity settings.

## Test workflow

* During implementation, run the smallest relevant test file or test selection first. Run the complete suite at most once after the targeted tests pass.
* GUI tests must mock every modal dialog they can open. An unexpected modal must fail immediately instead of waiting for interactive input.
