#!/usr/bin/env bash
# Build the macOS application bundle. Run on macOS with the dev requirements
# installed (python -m pip install -r requirements-dev.txt).
set -euo pipefail
cd "$(dirname "$0")"

python -m PyInstaller \
    --noconfirm \
    --clean \
    --onefile \
    --windowed \
    --name RustPainter \
    --icon RustPainterIcon.png \
    --add-data "RustPainterIcon.png:." \
    --add-data "assets/ui:assets/ui" \
    --osx-bundle-identifier com.rustpainter.app \
    main.py

test -d dist/RustPainter.app || { echo "PyInstaller did not create dist/RustPainter.app" >&2; exit 1; }
echo "Built dist/RustPainter.app"

mkdir -p release
# ditto preserves the bundle structure and extended attributes that a plain
# zip tool can mangle.
ditto -c -k --keepParent dist/RustPainter.app release/RustPainter-macOS.zip
echo "Packaged release/RustPainter-macOS.zip"
