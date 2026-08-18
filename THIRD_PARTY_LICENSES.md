# Third-party licenses

RustPainter does not bundle third-party artwork. The icons and surface textures
under `assets/ui` are project artwork, baked from the source renders by
`tools/prepare_ui_assets.py`.

Runtime dependencies are installed from PyPI rather than vendored into this
repository, and each ships under its own license:

| Package  | License      |
| -------- | ------------ |
| PySide6  | LGPL v3 / commercial (Qt for Python) |
| Pillow   | MIT-CMU      |
| NumPy    | BSD 3-Clause |

Builds produced by `build.ps1` bundle these dependencies into the executable.
If you redistribute such a build, review each project's terms — in particular
Qt's LGPL relinking requirements — and include their license texts alongside it.
