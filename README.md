# RustPainter

RustPainter is a local Windows 10/11 desktop utility that recreates an image in Rust's sign-painting UI by using ordinary screen coordinates and Windows mouse input. It does **not** read game memory, inject code, hook graphics, modify game files, or attempt to bypass anti-cheat.

The application keeps all profiles and settings on your PC. Every coordinate used for painting comes from a rectangle that you calibrate on your own display.

## Install

**You need:** Windows 10/11, and Rust running in borderless or windowed mode
(not exclusive fullscreen).

### Download the app (easiest)

Grab **`RustPainter.exe`** from the
[latest release](https://github.com/YeheyaMohammad01/RustPainter/releases/latest)
and double-click it. That single file is the whole application - there is no
installer, no Python to set up, and no admin rights needed. To uninstall,
delete the file.

> **Windows will warn you the first time.** The executable is not code-signed,
> so SmartScreen shows "Windows protected your PC". Click **More info** ->
> **Run anyway**. Some antivirus tools also flag it, because it is an unsigned
> program that synthesizes mouse input - exactly the shape of thing they watch
> for. Every release is built from this repository by
> [GitHub Actions](.github/workflows/release.yml), and each one ships a
> `.sha256` file you can check with
> `Get-FileHash RustPainter.exe -Algorithm SHA256`. If you would rather not
> trust a prebuilt binary, build your own with the steps below.

Want Start-menu and desktop shortcuts? Clone the repo and point
[`install.ps1`](install.ps1) at your download. It copies the executable to
`%LOCALAPPDATA%\Programs\RustPainter` and creates the shortcuts:

```powershell
.\install.ps1 -ExecutablePath $env:USERPROFILE\Downloads\RustPainter.exe
```

### Run from source

Requires 64-bit Python 3.11-3.14:

```powershell
git clone https://github.com/YeheyaMohammad01/RustPainter.git
cd RustPainter
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

## Quick start

Paint a sign in six steps:

1. **Open the sign** in Rust and leave its painting interface stationary.
2. **Calibrate.** In RustPainter, drag a box around each of three regions on
   your screen: the sign **canvas**, the **color box** (the large
   white-to-color-to-black square), and the **hue bar** (the narrow rainbow
   strip). Drag just inside each region - do not include its border.
3. **Load an image.** Browse or drag & drop. The defaults are ready to use.
4. **Check the preview.** The paint simulation shows what will be painted,
   plus the stroke count and estimated time.
5. **Dry run first.** Tick **Dry run**, press **F8**, and focus Rust during the
   countdown. Progress should complete without any clicks reaching the game.
6. **Paint.** Untick Dry run and start with a small, low-color test image.

**Hotkeys:** `F8` start/resume, `F9` pause, `F10` abort. Abort immediately
releases any held mouse button.

Keep a hand near **F10** during your first runs, and do not leave the tool
painting unattended. Everything below is detail you only need when tuning.

## Features

- Dark workspace built around the image -> calibrate -> paint flow, with advanced controls tucked into Settings
- Loads PNG, JPEG, WebP, BMP, TIFF and other Pillow-supported images by browse or drag & drop
- Fit, fill/crop, and stretch composition with a live paint simulation
- Adjustable painting resolution, palette size, dithering, transparency, and fit background
- Named profiles per sign/UI layout, each inheriting the current calibration
- Drag calibration for the canvas, color box, hue bar, and optional Size track and brush-preview tile, with an on-screen overlay to verify them
- Overpaint stroke merging that typically removes 10-40% of strokes without changing the finished image
- Speed presets (Relaxed / Standard / Fast / Turbo) over fully adjustable timing, with 1 ms Windows timer resolution while painting
- Per-profile color correction measured from a painted 32-swatch chart
- Safety throughout: countdown, dry run, foreground-window guard, corner abort, pause/resume, and an abort that always releases the mouse
- Local JSON profiles/settings and rotating logs; nothing leaves your PC

## Detailed setup and first paint

1. Run Rust in borderless or windowed mode for the easiest calibration and focus switching.
2. Open the target sign's painting interface and leave it stationary.
3. In RustPainter, create a profile for that sign/UI layout. A new profile starts with a copy of the current profile's calibration, so an unchanged setup needs no recalibration.
4. Calibrate the **canvas**, **color box**, and **hue bar**. Drag just inside each usable region; do not include borders. The color box is the large white/color/black square; the hue bar is only the narrow rainbow strip. Enable **Show calibration boxes on screen** to verify the stored rectangles as labeled red outlines over the game UI (they are click-through and hide automatically while painting).
5. If automatic brush sizing is wanted, also calibrate the clickable **Size track** and the separate gray **brush-preview tile** that displays the current brush footprint.
6. Load an image. The balanced defaults are ready to use; composition, quality, palette, background, and transparency controls are under **Settings → Artwork** when needed.
7. Inspect the paint simulation and plan statistics.
8. Enable **Dry run** for the first test. Press Start, focus Rust during the countdown, and confirm that progress completes without clicks.
9. Use the debug corner/center and color-picker tests, then paint one dot or short stroke.
10. Disable Dry run and begin a small, low-color test image. Keep the abort hotkey available.

The current Rust picker layout is fixed in the application: hue runs bottom to top, saturation increases left to right, brightness decreases top to bottom, and brush size increases left to right.

Default global hotkeys are **F8** start/resume, **F9** pause, and **F10** abort. Abort is designed to release any held mouse button immediately.

## Calibration and DPI notes

RustPainter opts into per-monitor DPI awareness before creating the GUI and stores virtual-desktop screen coordinates plus the current display layout with each profile. This supports Windows scaling and monitors left/above the primary display (negative coordinates). A display-layout warning means that the profile should be recalibrated before painting.

Exclusive fullscreen applications can prevent overlays, screenshots, focus checks, or synthetic input from behaving normally. Borderless fullscreen is recommended. If Rust or Windows is running at a different privilege level than RustPainter, Windows may reject input; run both at the same privilege level.

## Settings that require experimentation

Rust brush behavior can vary with sign type, selected in-game brush, frame rate, and UI scale. These controls are grouped under **Settings → Painting** so the main workspace stays focused:

- Painting speed preset (Relaxed / Standard / Fast / Turbo); editing any timing value switches it to Custom
- Stroke merging (Off / Balanced / Maximum) under Paint Quality
- Automatic Size-track search using the calibrated brush-preview footprint (optional)
- Logical pixel spacing
- Stroke duration/speed and interpolation step
- Mouse-down time for dots
- Picker and inter-stroke delays
- Canvas inset and logical resolution

Start with a low-resolution 8-color test. If adjacent rows bleed together, reduce the in-game brush size, increase logical spacing, or lower the logical resolution. If strokes have gaps, slow the stroke or reduce the interpolation step. Faster speed presets assume the game keeps up with rapid input; if paint goes missing, step back down to Standard.

**Stroke merging** exploits painting order: colors are painted from most to least frequent, so an earlier color may paint across pixels that a later color repaints anyway. The final image is identical, but fragmented regions (text backgrounds, dithered gradients) need far fewer mouse strokes. *Balanced* merges across gaps of up to 6 logical pixels and is almost always the fastest choice; *Maximum* produces the fewest strokes but can spend extra time traveling across very long overpainted spans. The paint-plan panel reports how many strokes merging removed.

Automatic brush sizing works with a solid square or circle. Before painting it temporarily selects a high-contrast color, measures the live preview, and searches the Size track for a footprint slightly smaller than one logical cell. Spray/noise brushes do not have a stable footprint and must be sized manually.

## Sign color correction

The Paint Simulation uses ordinary RGB, while Rust renders paint through the sign material, wood texture, and lighting. A profile can measure and compensate for that response:

1. Use a blank/reset disposable sign with the profile fully calibrated.
2. Under **Settings → Color**, click **Prepare Calibration Chart**. This replaces the imported image and selects Stretch, Very Fast, 32 colors, and no dithering.
3. Paint the complete chart through the normal Start/countdown workflow. Any older correction is automatically bypassed for this chart.
4. Leave the finished chart visible in Rust and click **Measure Painted Chart**.
5. Focus Rust during the capture countdown. The app samples 32 large swatches, rejects inconsistent captures, and saves the measured correction to that profile.
6. Reset/use a fresh sign and reload the artwork. Correction is applied automatically to future paint jobs.

The chart deliberately consumes paint on one test sign. Re-measure after changing sign material, display/graphics color behavior, or the main canvas/picker calibration. **Clear Color Correction** restores direct RGB-to-picker mapping.

## Safety behavior

- Starting uses a visible countdown so you can focus Rust.
- With the foreground guard enabled, every populated selector must match: the configured window-title fragment and, when supplied, the executable name. Loss of focus pauses and releases the mouse.
- F9 pauses at the next short cancellation checkpoint.
- F10 aborts, clears pending work, and releases the mouse.
- Optional rapid movement into a virtual-screen corner aborts the job.
- UI-reference comparison is only a coarse warning for moved/wrong UI; it is not computer vision and should not replace calibration checks.

Keep your hand near F10 during initial tests. Do not use the tool unattended.

## Build a standalone executable

Install the development requirements, then run:

```powershell
.\build.ps1
```

The self-contained executable is created at `dist\RustPainter.exe`, with the RustPainter icon embedded. It can be copied to the desktop or another folder and opened by itself; no `_internal` directory or separate Python installation is required. A one-file build can take a little longer to start because it extracts its bundled runtime to a temporary directory. Windows or antivirus software may warn about an unsigned locally built executable that generates input; review and build the source yourself.

The build also creates a distributable archive at `release\RustPainter-Windows-x64.zip`. Users can unzip it anywhere and run the single executable inside.

To install the current build for your Windows account and create clean `RustPainter` shortcuts on the desktop and Start menu, run:

```powershell
.\install.ps1
```

The actual executable is installed under `%LOCALAPPDATA%\Programs\RustPainter`; the desktop item is a shortcut, so Windows does not display an `.exe` suffix.

If PowerShell blocks local scripts under its execution policy, run the build once with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\build.ps1
```

### Cutting a release

Releases are built and published by
[`.github/workflows/release.yml`](.github/workflows/release.yml). Pushing a
version tag builds the executable on a clean Windows runner, runs the test
suite, and attaches `RustPainter.exe` plus its SHA256 checksum to a new GitHub
release:

```powershell
git tag v1.0.0
git push origin v1.0.0
```

You can also run the workflow by hand from the Actions tab to get a test build
as a downloadable artifact without tagging anything.

## Local data and logs

Profiles, settings, calibration reference captures, and logs are stored under `%LOCALAPPDATA%\RustPainter` when available. Deleting that folder resets the application; exporting or copying its JSON files is enough to back up calibration data.

## Known limitations

- The app cannot know Rust's internal brush radius, native sign resolution, or exact picker gradient. Calibration and small test strokes are required.
- `SendInput` may be ignored by exclusive fullscreen, elevated, protected, or anti-cheat-managed windows. The utility does not work around those restrictions.
- Horizontal runs are deliberately prioritized for reliability; complex images can still require many strokes.
- Color accuracy is approximate because the displayed picker, monitor color, sign material, lighting, and in-game rendering can alter the result.
- Measured correction compensates the average captured material response, but cannot remove physical plank seams, spatial lighting variation, or colors outside Rust's available gamut.
- Simple reference-image comparison catches large layout changes but not every wrong state.

## Project layout

```text
main.py
app/
  gui/                 PySide6 interface
  calibration.py       full-screen rectangle selector
  profiles.py          profile JSON persistence
  settings.py          defaults and local settings
  image_processing.py  composition and palette reduction
  color_mapping.py     RGB/HSV picker mapping
  brush_calibration.py live brush-preview measurement
  color_calibration.py painted-chart response fitting
  coordinates.py       logical/screen coordinate conversion
  paint_plan.py        horizontal-run planning and estimates
  input_controller.py  SendInput and dry-run input
  hotkeys.py           Windows global hotkeys
  painter.py           resumable, abortable execution
  screen.py            DPI/display/focus/capture helpers
tests/
```

## Contributing

Issues and pull requests are welcome. Please run the test suite before opening a
pull request:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
```

The GUI tests use `pytest-qt`. If you are running headless, set
`QT_QPA_PLATFORM=offscreen`.

## Disclaimer

RustPainter is an independent, unofficial project. It is not affiliated with,
endorsed by, or sponsored by Facepunch Studios. "Rust" is a trademark of its
respective owner and is used here only to describe what the tool interoperates
with.

The tool automates ordinary mouse input against your own display. It does not
read game memory, inject code, hook graphics, modify game files, or attempt to
bypass anti-cheat. Even so, automating input may violate the terms of service
or server rules of the software you point it at. You are responsible for
deciding whether your use is permitted, and you use this software at your own
risk. See the warranty disclaimer in [LICENSE](LICENSE).

## License

Released under the [MIT License](LICENSE). Third-party runtime dependencies
ship under their own terms; see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
