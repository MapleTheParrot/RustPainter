#!/usr/bin/env bash
# Build the macOS application bundle. Run on macOS with the dev requirements
# installed (python -m pip install -r requirements-dev.txt).
set -euo pipefail
cd "$(dirname "$0")"

# Stamped into the bundle so Finder and the Info pane show a real version.
VERSION="${RUSTPAINTER_VERSION:-$(git describe --tags --abbrev=0 2>/dev/null || true)}"
VERSION="${VERSION#v}"          # accept either v1.2.3 or 1.2.3
VERSION="${VERSION:-0.0.0}"

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

APP="dist/RustPainter.app"
test -d "$APP" || { echo "PyInstaller did not create $APP" >&2; exit 1; }

# Editing Info.plist invalidates the signature PyInstaller applied, so the
# bundle must be re-signed afterwards -- see the ad-hoc signing note below.
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $VERSION" "$APP/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Add :CFBundleVersion string $VERSION" "$APP/Contents/Info.plist" 2>/dev/null \
    || /usr/libexec/PlistBuddy -c "Set :CFBundleVersion $VERSION" "$APP/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Add :LSMinimumSystemVersion string 11.0" "$APP/Contents/Info.plist" 2>/dev/null \
    || /usr/libexec/PlistBuddy -c "Set :LSMinimumSystemVersion 11.0" "$APP/Contents/Info.plist"

# Ad-hoc signature ("-"). This is NOT a Developer ID signature and does not
# satisfy Gatekeeper -- users still have to approve the app once. It is
# required regardless: arm64 refuses to execute a binary with no signature at
# all. Verification is strict so a broken bundle fails here rather than on a
# user's Mac.
codesign --force --sign - --identifier com.rustpainter.app "$APP"
codesign --verify --strict --verbose=2 "$APP"

echo "Built $APP (version $VERSION)"

mkdir -p release
# ditto preserves the bundle structure, symlinks, and extended attributes that
# a plain zip tool can mangle.
ditto -c -k --keepParent "$APP" release/RustPainter-macOS.zip
echo "Packaged release/RustPainter-macOS.zip"
