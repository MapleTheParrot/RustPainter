# RustPainter

RustPainter is a local desktop utility for Windows 10/11 and macOS that recreates an image in Rust's sign-painting UI by using ordinary screen coordinates and synthesized mouse input. It does **not** read game memory, inject code, hook graphics, modify game files, or attempt to bypass anti-cheat.

The application keeps all profiles and settings on your PC. Every coordinate used for painting comes from a rectangle that you calibrate on your own display.

## Install

**You need:** Windows 10/11 or macOS, and Rust running in borderless or
windowed mode (not exclusive fullscreen).

### Download the app (easiest)

**Windows:** grab **`RustPainter.exe`** from the
[latest release](https://github.com/YeheyaMohammad01/RustPainter/releases/latest)
and double-click it. That single file is the whole application - there is no
installer, no Python to set up, and no admin rights needed. To uninstall,
delete the file.

**macOS:** from the same release, grab **`RustPainter-macOS-arm64.zip`** on
Apple Silicon (M1 and later) or **`RustPainter-macOS-x86_64.zip`** on an Intel
Mac - check the Apple menu > About This Mac if unsure. Requires macOS 11 Big
Sur or newer. Unzip it and move `RustPainter.app` to Applications.

The app is **ad-hoc signed but not notarized**, because notarizing requires a
paid Apple Developer account. macOS will therefore block it once, and you have
to approve it by hand:

1. Double-click the app. macOS refuses to open it.
2. Open **System Settings > Privacy & Security**, scroll to the **Security**
   section near the bottom, and click **Open Anyway** next to RustPainter.
3. Confirm with **Open**.

Control-clicking the app and choosing *Open* used to work for this and
**no longer does** - Apple removed that shortcut in macOS 15 Sequoia. If you
prefer the terminal, this achieves the same thing in one step:

```bash
xattr -dr com.apple.quarantine /Applications/RustPainter.app
```

Then grant two permissions under **System Settings > Privacy & Security**:

- **Accessibility** - required to move the mouse and for the global
  F8/F9/F10 hotkeys.
- **Screen Recording** - required for brush and color calibration captures.

Relaunch the app after granting them.

macOS throttles background applications through App Nap, which would slow a
paint job down for as long as the game is frontmost. RustPainter opts out of it
automatically for the duration of each job, so no configuration is needed.

> **After installing a new version**, macOS may stop honouring permissions you
> granted to the previous build. An ad-hoc signature changes every time the app
> is rebuilt, and the permission system tracks apps by signature. If hotkeys or
> calibration capture stop working after an update, remove RustPainter from the
> Accessibility and Screen Recording lists (select it, press **-**) and add it
> again.

If your keyboard uses F8-F10 as media keys, either hold **Fn** when pressing a
hotkey or enable *Use F1, F2, etc. keys as standard function keys* under
System Settings > Keyboard.

### Run from source

Requires 64-bit Python 3.11-3.14. On Windows:

```powershell
git clone https://github.com/YeheyaMohammad01/RustPainter.git
cd RustPainter
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

On macOS (the pyobjc frameworks install automatically from requirements.txt):

```bash
git clone https://github.com/YeheyaMohammad01/RustPainter.git
cd RustPainter
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

## Quick start

Paint a sign in five steps:

1. **Open the sign** in Rust and leave its painting interface stationary.
2. **Calibrate.** In RustPainter, drag a box around each of three regions on
   your screen: the sign **canvas**, the **color box** (the large
   white-to-color-to-black square), and the **hue bar** (the narrow rainbow
   strip). Drag just inside each region - do not include its border.
3. **Load an image.** Click the preview, use **Choose image**, or drag & drop
   a file anywhere in the window. The defaults are ready to use.
4. **Check the preview.** The paint simulation shows what will be painted,
   plus the stroke count and estimated time.
5. **Paint.** Start with a small, low-color test image and focus Rust during
   the countdown.

**Hotkeys:** `F8` start/resume, `F9` pause, `F10` abort. Abort immediately
releases any held mouse button.

Keep a hand near **F10** during your first runs, and do not leave the tool
painting unattended. Everything below is detail you only need when tuning.

## Features

- Dark, rust-textured workspace built around the image -> calibrate -> paint flow, with advanced controls tucked into Settings
- Loads PNG, JPEG, WebP, BMP, TIFF and other Pillow-supported images by clicking either preview, browsing, or dropping a file anywhere in the window
- Fit, fill/crop, and stretch composition with a live paint simulation
- Adjustable painting resolution, palette size, dithering, transparency, and fit background
- One-click background removal that leaves a plain backdrop unpainted, by detected or picked color, with an adjustable tolerance
- Multiple draggable text layers with inline editing, resize handles, Ctrl+D or Ctrl+C to copy the selected layer, Delete to remove it, and live font/color styling
- Text sized as a fraction of the canvas, so a caption keeps its proportions when the quality preset changes the painting resolution
- Named profiles per sign/UI layout, each inheriting the current calibration
- Drag calibration for the canvas, color box, hue bar, and optional Size track, brush-preview tile, and Square/Circle shape buttons, with an on-screen overlay to verify them
- Optimization modes (Exact / Quality / Balanced / Fast) that plan like a painter: perceptually identical colors merge, insignificant specks are absorbed, and large areas are filled with the largest safe brush before details go on top, with the preview showing exactly what will be painted
- Overpaint stroke merging that typically removes 10-40% of strokes without changing the finished image
- Speed presets (Relaxed / Standard / Fast / Turbo) over fully adjustable timing, with 1 ms Windows timer resolution while painting
- Per-profile color correction measured from a painted 32-swatch chart, and per-profile brush sizing measured from dabs painted on the canvas itself
- Safety throughout: countdown, foreground-window guard, auto-pause when you move the mouse, corner abort, pause/resume, and an abort that always releases the mouse
- Local JSON profiles/settings and rotating logs; nothing leaves your PC

## Detailed setup and first paint

1. Run Rust in borderless or windowed mode for the easiest calibration and focus switching.
2. Open the target sign's painting interface and leave it stationary.
3. In RustPainter, create a profile for that sign/UI layout. A new profile starts with a copy of the current profile's calibration, so an unchanged setup needs no recalibration.
4. Calibrate the **canvas**, **color box**, and **hue bar**. Aim just inside each usable region; a pixel of overshoot is corrected automatically (see below). The color box is the large white/color/black square; the hue bar is only the narrow rainbow strip. Enable **Show calibration boxes on screen** to verify the stored rectangles as labeled red outlines over the game UI (they are click-through and hide automatically while painting).
5. If automatic brush sizing is wanted, also calibrate the clickable **Size track** and the separate gray **brush-preview tile** that displays the current brush footprint.
6. Load an image. The balanced defaults are ready to use; composition, quality, palette, background, and transparency controls are under **Settings → Artwork** when needed.
7. Inspect the paint simulation and plan statistics.
8. Use the debug corner/center and color-picker tests, then paint one dot or short stroke.
9. Begin with a small, low-color test image. Keep the abort hotkey available.

The niche **Run without mouse input** diagnostic remains available under
**Settings → Diagnostics** for plan timing and troubleshooting, but is off by default.

### Skipping a background you do not want painted

A plain backdrop is usually most of the strokes in a sign. **Remove background**
in the workspace's quick settings leaves it unpainted, so Rust paints only the
subject and the plan gets shorter by exactly those pixels.

- **Background** - *Detect from the edges* votes on the colors ringing the
  artwork, which suits a product shot or a logo on a flat field. Switch to
  *Pick a color* when the subject reaches the edges or a specific color should go.
- **Tolerance** - how far a pixel may drift from that color and still be
  skipped. Raise it for photographs and JPEG artifacts; lower it when part of
  the subject starts disappearing.
- **Touching the edges / Anywhere in the image** - edge matching only removes
  background reachable from outside the artwork, so enclosed areas (the hole in
  an O, a white eye) keep their paint. *Anywhere* also removes every matching
  pocket inside the subject.

Removal happens before the palette is chosen, so a skipped backdrop no longer
consumes one of the requested colors. Text layers still paint over removed
areas.

### Optimization modes

The **Optimization** picker in quick settings chooses how boldly planning may
simplify the image to paint faster. The Rust preview always shows the actual
image the plan will reproduce, so the trade-off is visible before painting.

- **Exact** - the classic plan: every quantized pixel, row by row. Use it when
  pixel-level fidelity matters more than speed. Stroke merging stays available
  in this mode only; the optimized modes handle it themselves.
- **Quality** - very conservative: merges only colors that are genuinely
  indistinguishable and cleans up single-pixel noise.
- **Balanced** (recommended) - merges visually identical colors, absorbs
  insignificant specks into their surroundings, and paints large regions first
  with the largest safe brush.
- **Fast** - aggressive merging and cleanup for the shortest paint time,
  usually still within a few percent of the source's look.

Optimized modes paint most-common colors first and may sweep straight across
pixels a later color repaints anyway - the same idea as stroke merging, taken
further. With **Automatic brush sizing** calibrated they also fill wide areas
with a bigger brush before switching down for edges and detail; a resize costs
real seconds, so the planner only fetches a big brush when it pays for the trip.
Dithered images keep their deliberate speckle: region cleanup turns itself off
when dithering is enabled.

Two further optional calibrations, **Square shape** and **Circle shape**, mark
Rust's solid brush-shape buttons. With neither calibrated, painting keeps
whatever shape is selected in Rust. With one, optimized plans may select that
shape when a large brush is used. With both, the planner picks square or circle
per region - square for broad flat fills, circle where a square would spill
into places that need cleaning up - and batches work so it rarely switches.

### How the image is prepared for the picker

Two steps of the pipeline exist purely to keep the sign faithful to the source:

- **Resampling runs in linear light.** Averaging gamma-encoded sRGB weighs
  perceptual codes instead of photons, so a phone photo reduced to a few
  hundred pixels lands roughly twenty RGB levels dark. Decoding before the
  resize and re-encoding after keeps the shrunken image as bright as what you
  imported.
- **Near-neutral colors snap to gray.** Hue is meaningless below about four
  percent saturation - five levels of channel spread on a near-white pixel
  swing it by nearly 180 degrees - but the picker still gets a fully saturated
  hue click that only a click a pixel or two from the edge of the saturation
  box pulls back. Any slack in that calibration used to paint pastel speckle
  across a white backdrop. Snapping also stops several indistinguishable
  near-whites from each buying a palette entry the artwork could use.

The current Rust picker layout is fixed in the application: hue runs bottom to top, saturation increases left to right, brightness decreases top to bottom, and brush size increases left to right.

Default global hotkeys are **F8** start/resume, **F9** pause, and **F10** abort. Abort is designed to release any held mouse button immediately.

## Calibration and DPI notes

Before the first stroke of every job, the calibrated color box and hue bar are
captured and shrunk to the widget Rust is actually drawing inside them. This
matters more than it sounds: the mapping sends saturation 0, saturation 1,
value 0, value 1, and hue 0 degrees to the exact edges of the stored rectangle,
so overshooting a hue bar by one pixel puts every one of those clicks on the
panel behind it. Rust ignores such a click, the color silently stays whatever
was selected before, and grays, whites, blacks, and pure reds paint in a
leftover color. Measuring at paint time rather than at calibration time means
profiles already on disk are corrected without recalibrating, and a picker that
shifted by a pixel is re-measured on the next run. A capture that does not
clearly show the picker leaves the calibration exactly as you drew it.


RustPainter opts into per-monitor DPI awareness before creating the GUI and stores virtual-desktop screen coordinates plus the current display layout with each profile. This supports Windows scaling and monitors left/above the primary display (negative coordinates). A display-layout warning means that the profile should be recalibrated before painting.

On macOS the app works in global display points, which Qt, Quartz input events, and the calibration overlay all share, so Retina scaling needs no special handling; screen captures are normalized from pixel scale automatically. Synthetic input and hotkeys stop working if the Accessibility permission is revoked, and calibration captures require Screen Recording.

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

Automatic brush sizing works with a solid square or circle. Spray/noise brushes do not have a stable footprint and must be sized manually.

### Measuring the brush on the canvas

The brush-preview tile draws the brush at the tile's own scale, which is not the
canvas's scale. A footprint measured there therefore answers a different
question than the planner asks - how many canvas pixels will this brush cover -
so cells sized from it come out too large and bleed into their neighbours. This
shows up worst on low-resolution pixel art, where the cells are big and a brush
one step too wide smears several of them together.

**Measure Brush on Canvas**, next to Automatic brush sizing, settles it by
measuring what Rust actually paints:

1. Use a blank or disposable sign with the profile's canvas, picker, and Size
   track calibrated.
2. Click **Measure Brush on Canvas** and focus Rust for the countdown.
3. The painter primes six small squares of canvas with the widest brush, stamps
   one dab in each at a different Size-track position, and reads back how wide
   each dab came out. The probes crowd the low end of the track, where a
   mis-sized brush does the most damage.
4. The measured curve is stored on the profile.

Painting then sets the brush straight from that curve. Nothing has to be
measured mid-job, so a run never pauses to hunt the slider, and the measurement
includes the brush's soft edge rather than modelling it. **Clear Measurement**
returns to the preview-tile search. Re-measure after changing sign type, brush
shape, or the canvas calibration.

## Sign color correction

The Paint Simulation uses ordinary RGB, while Rust renders paint through the sign material, wood texture, and lighting. A profile can measure and compensate for that response:

1. Use a blank/reset disposable sign with the profile fully calibrated.
2. Under **Settings → Color**, click **Prepare Calibration Chart**. This replaces the imported image and selects Stretch, Very Fast, 32 colors, and no dithering.
3. Paint the complete chart through the normal Start/countdown workflow. Any older correction is automatically bypassed for this chart.
4. Leave the finished chart visible in Rust and click **Measure Painted Chart**.
5. Focus Rust during the capture countdown. The app samples 32 large swatches, rejects inconsistent captures, and saves the measured correction to that profile.
6. Reset/use a fresh sign and reload the artwork. Correction is applied automatically to future paint jobs.

Once measured, the Rust preview renders artwork through the model as well, so the preview and the sign agree. Colors the material can reach look unchanged; colors outside its measured gamut show as the muted version Rust will actually produce instead of a promise it cannot keep.

The chart deliberately consumes paint on one test sign. Re-measure after changing sign material, display/graphics color behavior, or the main canvas/picker calibration. **Clear Color Correction** restores direct RGB-to-picker mapping.

## Safety behavior

- Starting uses a visible countdown so you can focus Rust.
- With the foreground guard enabled, every populated selector must match: the configured window-title fragment and, when supplied, the executable name. On Windows the expected process defaults to `RustClient.exe`; on macOS it defaults to empty so the window title governs, because a Windows executable name can never match there and would pause the job the moment you focused the game. If you upgraded from an earlier build and the job pauses immediately after the countdown, clear **Expected process name** under Settings. Reading the frontmost window title needs Screen Recording on macOS (otherwise the application name is matched instead). Loss of focus pauses and releases the mouse.
- F9 pauses at the next short cancellation checkpoint.
- F10 aborts, clears pending work, and releases the mouse.
- **Mouse guard** (on by default, Settings > Safety): taking the mouse back mid-job pauses it instead of letting your hand fight the painter. The button is released within milliseconds, and resuming with F8 re-selects the color and repeats the interrupted stroke, so a bump costs one stroke rather than the whole sign. Movement is detected from the gap between where the painter put the cursor and where the cursor actually is; no input hook is installed. If a job pauses repeatedly on its own, the calibrated canvas is likely outside where the game lets the cursor go - recalibrate, or turn the guard off.
- Optional rapid movement into a virtual-screen corner aborts the job. The corner stop keeps working while a job is paused, so the same gesture still ends a run the mouse guard has already halted.
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

- The app cannot know Rust's internal brush radius, native sign resolution, or exact picker gradient. Calibration and small test strokes are required, and the brush is best measured on the canvas rather than read off the preview tile.
- `SendInput` may be ignored by exclusive fullscreen, elevated, protected, or anti-cheat-managed windows. The utility does not work around those restrictions.
- Horizontal runs are deliberately prioritized for reliability; complex images can still require many strokes.
- Color accuracy is approximate because the displayed picker, monitor color, sign material, lighting, and in-game rendering can alter the result. Without a measured correction the preview shows the commanded RGB, which is what the picker is asked for and not what the lit sign returns.
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
  image_processing.py  composition, background removal, palette reduction
  color_mapping.py     RGB/HSV picker mapping
  brush_calibration.py live brush-preview measurement
  color_calibration.py painted-chart response fitting
  coordinates.py       logical/screen coordinate conversion
  paint_plan.py        horizontal-run planning and estimates
  paint_optimizer.py   artist-style optimized planning (modes, brushes)
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
