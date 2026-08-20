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
- Fit, fill/crop, and stretch composition with a live paint simulation, with the Fill crop draggable directly on the source image so the sign keeps the part of the picture you meant
- Adjustable painting resolution, palette size, dithering, transparency, and fit background
- One-click background removal that leaves the backdrop unpainted, by detected or picked color, with a smart matcher for gradients, vignettes and photographic backdrops that also clears the halo off a cut-out subject
- Multiple draggable text layers edited right on the Source tab - inline editing, resize handles, Ctrl+D or Ctrl+C to copy, Delete to remove, and a bracketed outline showing the part of the image the sign will hold - while the Rust preview shows the text baked in exactly as it will paint, marks itself read-only, and offers a way back to the Source tab if you try to edit there
- Text editing conveniences you would expect from a graphics app: select several layers with a rubber band, Shift+click, Ctrl+click or Ctrl+A and restyle or drag them together, snap to the middle and edges of the sign and to the other layers while dragging (Alt bypasses), align and spread buttons, and arrow-key nudging
- Per-layer gradients and outlines, both drawn by the same renderer the paint plan bakes, so the letters on the Source tab are the letters the sign receives
- Undo and redo over the text layers alone (Ctrl+Z / Ctrl+Y anywhere in the window, mid-typing included), so a whole drag or a run of keystrokes steps back as one edit
- Text sized as a fraction of the canvas, so a caption keeps its proportions when the quality preset changes the painting resolution
- Named profiles per sign/UI layout, each inheriting the current calibration
- Drag calibration for the canvas, color box, hue bar, and - for automatic brush sizing - the numeric Size field and Rust's clear button, with an on-screen overlay to verify them
- Brush sizing measured from the sign itself at the start of every job: a few probe strokes fit what Rust's Size numbers really cover, the sign is wiped clean again, and only then does the artwork go down
- Optimization modes (Exact / Quality / Balanced / Fast) that plan like a painter: perceptually identical colors merge, insignificant specks are absorbed, and large areas are filled with the largest safe brush before details go on top, with the preview showing exactly what will be painted
- Overpaint stroke merging that typically removes 10-40% of strokes without changing the finished image
- Speed presets (Relaxed / Standard / Fast / Turbo) over fully adjustable timing, with 1 ms Windows timer resolution while painting
- Per-profile color correction measured from a painted 32-swatch chart
- A Timelapse tab that captures a PNG frame of the sign at a set interval while painting, plays a recording back inside the app, and saves it as a video file - no external encoder required, and playback speed is a slider that says how much of the paint job one second of video covers
- Safety throughout: countdown, foreground-window guard, auto-pause when you move the mouse, corner abort, pause/resume, and an abort that always releases the mouse
- Plans already computed are kept, so stepping back to a preset you already tried comes back instantly instead of being recalculated, and any recalculation that does run covers the preview with what it is working on
- Local JSON profiles/settings and rotating logs; nothing leaves your PC

## Detailed setup and first paint

1. Run Rust in borderless or windowed mode for the easiest calibration and focus switching.
2. Open the target sign's painting interface and leave it stationary.
3. In RustPainter, create a profile for that sign/UI layout. A new profile starts with a copy of the current profile's calibration, so an unchanged setup needs no recalibration.
4. Calibrate the **canvas**, **color box**, and **hue bar**. Aim just inside each usable region; a pixel of overshoot is corrected automatically (see below). The color box is the large white/color/black square; the hue bar is only the narrow rainbow strip. Enable **Show calibration boxes on screen** to verify the stored rectangles as labeled red outlines over the game UI (they are click-through, hide themselves while a job is actually painting, and come back while it is paused so the boxes can be checked against Rust before letting it carry on).
5. If automatic brush sizing is wanted, calibrate the numeric **Size value box** beside Rust's size slider and the **Clear button** (Rust's trash icon, which wipes the sign). There is nothing to run by hand: every paint job measures the brush itself, then clears the sign before it paints. Measuring paints a scout stroke to find the scale, then a few probe strokes bracketing the brush your resolution needs, and reads back how much of the sign each one actually covered. Only solid coverage counts - Rust's brush fades out over its last texture pixel, and a cell left under that fade still looks unpainted. Measuring every run rather than once is what removes the step that used to be missed: a stored measurement describes a sign you may since have re-framed, walked away from, or replaced, and a stale one paints the whole image at the wrong width.
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
- **Smart / Touching the edges / Anywhere in the image** - how the match is
  made, described below.

*Touching the edges* and *Anywhere in the image* both match one flat color.
Edge matching only removes background reachable from outside the artwork, so
enclosed areas (the hole in an O, a white eye) keep their paint; *Anywhere*
also removes every matching pocket inside the subject.

**Smart** is the default, and is for backdrops that are not one flat color -
which is most of them. A studio sweep is a gradient, a photograph has a
vignette, and a JPEG has ringing along every edge, so one key color matches the
middle of the range and neither end. Smart works differently in three ways:

- It reads **several** background colors off a band around the artwork instead
  of one off a single-pixel ring, and measures each pixel against the nearest.
- The tolerance decides only where the background *certainly* is. From those
  seeds the region spreads outwards through anything merely plausible, and only
  through pixels reachable from outside the artwork - so a gradient comes away
  end to end without the tolerance ever having to be wide enough to reach the
  subject, and enclosed pockets still keep their paint.
- The last pixel or two before the background is a blend of the two and matches
  neither, which is what leaves an outline around a cut-out subject. Those get a
  much looser match, but only where they are genuinely nearer the backdrop than
  the artwork behind them is - so a blend goes and the edge of a merely pale
  subject stays.

Because Smart grows past the tolerance rather than stopping at it, it reaches
further than the same number would elsewhere. If it starts taking the subject,
lower the tolerance rather than raising it.

Removal happens before the palette is chosen, so a skipped backdrop no longer
consumes one of the requested colors. Text layers still paint over removed
areas.

### Saving the Rust preview

The button beside the **PREVIEW** heading writes the Rust preview to an image
file. It is saved at one pixel per logical cell - the grid the plan is
expressed in - so a screenshot of the finished sign scaled down to the same
size lines up cell for cell and the difference between what was promised and
what Rust produced is a straight subtraction. Formats that carry alpha (PNG,
WebP) leave the unpainted cells empty instead of drawing the checkerboard the
preview shows them over; JPEG and BMP get the checker, having nowhere else to
put it. When the profile carries a measured color correction, the file shows
the colors the sign is predicted to return, exactly as the preview does.

### Transparency and soft edges

Painting has no alpha channel, so every pixel of the source has to become one
solid color or none at all. **Settings → Artwork** decides both halves of that:
**Transparent pixels** covers the fully transparent ones, and **Alpha fill**
covers the ones in between - the anti-aliased rim of a logo, the feathered
outline of a cut-out subject.

Alpha fill is off by default. Off, a pixel is painted only once it is more
opaque than not, and it is painted in its own color, so nothing appears on the
sign that the artwork did not contain. On, partial alpha is mixed into the
background color the way a compositor would, which is right when the sign
really will carry that background behind the artwork - and wrong otherwise,
because a feathered edge then paints a halo of half-background around the
subject.

### Laying out text

Every text layer lives on the Source tab, where it is dragged, resized by its
handles, and edited by double-clicking it. The layer picked in the side panel
owns the **Text** field; everything else - font, size, color, style, gradient,
outline - is written to the whole selection at once.

- **Selecting several layers** - drag a box across bare canvas to sweep up the
  layers it touches, Shift+click or Ctrl+click to add or drop one, or Ctrl+A to
  take them all. Holding either modifier while dragging a box widens the
  selection rather than starting a new one. Escape lets go. A group drags as
  one, so the layers keep their spacing.
- **Snapping** - a dragged layer jumps onto the middle of the sign, its edges,
  and the centre lines and edges of the other layers when it comes within a few
  pixels of them; a line marks whatever it caught. Hold **Alt** while dragging
  to place it by hand instead.
- **Align and Spread** - *Left / Center / Right / Top / Middle / Bottom* park
  the selection against an edge or midline of the sign. *Across* and *Down*
  leave an equal gap between three or more selected layers.
- **Arrow keys** nudge the selection one logical canvas pixel at a time, ten
  with Shift held.
- **Undo and Redo** cover the text layers only, not the rest of the settings.
  Ctrl+Z and Ctrl+Y (or Ctrl+Shift+Z) do the same from anywhere in the window -
  over the canvas, in the side panel's fields, and while a layer is being typed
  into. A whole drag, or a run of keystrokes, steps back as one edit, and
  opening a saved settings document starts the history over.
- **Gradient** fades the letters from the text color into a second one, down,
  across, or diagonally. The fade is quantized with the rest of the artwork, so
  a narrow palette shows it as bands rather than as a smooth ramp.
- **Outline** rings every letter with that many logical pixels of a chosen
  color, which is what keeps a caption readable over artwork it shares a color
  with. It paints extra cells, so it costs a little time.

Sizes are stored as a fraction of the canvas height rather than in pixels, so
captions keep their proportions when a quality preset changes the painting
resolution.

Under **Stretch**, the Source tab shows the image already stretched to the
shape of the sign rather than at its own proportions. Stretch is the one mode
that reshapes the artwork, so laying text out over the undistorted original
would put every caption somewhere the sign never had it; pre-distorting the
backdrop makes the tab agree with what gets painted. Fit and Fill leave the
source's shape alone and mark the sign's own area with a bracketed outline
instead.

### Framing a Fill crop

Under **Fill / Crop** the sign keeps only the part of the image that fits its
shape, and the Source tab shows which part: the kept region is bracketed and
everything it drops is dimmed. Drag anywhere on bare canvas to move that frame
onto what you want the sign to hold - the pointer becomes an open hand wherever
there is room to move - and **Crop alignment** switches to *Custom - dragged*.
Picking a named anchor again puts the crop back on it. Text layers are anchored
to the sign rather than to the image, so they travel with the frame.

While a Fill crop can move, gathering several text layers with a rubber band
takes Shift+drag or Ctrl+drag; a plain drag is the crop.

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
further. With **Automatic brush sizing** measured they also fill wide areas
with a bigger brush before switching down for edges and detail; a resize still
costs a click and a typed number, so the planner only fetches a big brush when
it pays for the trip. The measurement also tells the planner the widest brush
Rust can actually reach on that sign, so it never plans a pass the Size field
would have to clamp.
Dithered images keep their deliberate speckle: region cleanup turns itself off
when dithering is enabled.

Painting always keeps whatever brush shape is selected in Rust; the planner
budgets every pass for the square brush's worst-case spill, so either solid
shape stays safe.

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

Compact laptop keyboards often put F5-F12 behind an **Fn** key that the keyboard
firmware handles itself, so those presses never reach the app and the hotkeys
appear dead. Settings > Safety and hotkeys also offers **Ctrl+Alt+S / P / X / B /
N / M**, which any keyboard can produce and which Rust does not bind.

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
- Automatic brush sizing, which types the Size number a logical cell needs (optional; needs the Size value box and the clear button calibrated)
- Logical pixel spacing
- Stroke duration/speed and interpolation step
- Mouse-down time for dots
- Picker and inter-stroke delays
- Canvas inset and logical resolution

Start with a low-resolution 8-color test. If adjacent rows bleed together, reduce the in-game brush size, increase logical spacing, or lower the logical resolution. If strokes have gaps, slow the stroke or reduce the interpolation step. Faster speed presets assume the game keeps up with rapid input; if paint goes missing, step back down to Standard.

**Stroke merging** exploits painting order: colors are painted from most to least frequent, so an earlier color may paint across pixels that a later color repaints anyway. The final image is identical, but fragmented regions (text backgrounds, dithered gradients) need far fewer mouse strokes. *Balanced* merges across gaps of up to 6 logical pixels and is almost always the fastest choice; *Maximum* produces the fewest strokes but can spend extra time traveling across very long overpainted spans. The paint-plan panel reports how many strokes merging removed.

Automatic brush sizing works with a solid square or circle. Spray/noise
brushes do not have a stable footprint and must be sized manually.

Because the brush is measured on the sign rather than inferred from Rust's
preview tile, an impossible plan is refused before the first stroke instead of
discovered as smearing. If the smallest brush Rust offers is wider than one
logical cell, RustPainter says so and names the resolution that would fit -
that limit is the sign's own texture resolution, which no setting can raise.

A one-cell brush targets the full logical cell plus half a sign texel: the
sign renders strokes snapped to its own texture rows, and a brush sized
exactly to the row pitch still lands half a texel narrow wherever snapping
rounds down - which shows as bare stripes across the painting. The half-texel
overlap is invisible instead: boundaries are texel-quantized either way, and
the later-painted color owns the shared texel. Perfectly edge-to-edge pixels
still favor the **square** brush: a circle of one cell across leaves gaps at
the cell corners. Along a horizontal run the two shapes behave alike; the
difference shows at run ends and on isolated pixels.

## Timelapse

The **Timelapse** tab, next to Workspace in the header, captures the calibrated
canvas region at a regular interval while a job paints. Each paint job gets its
own timestamped folder of numbered PNG frames (``frame_00001.png``, …) under
the app's data directory. The interval is adjustable from 1 to 600 seconds,
paused time is skipped, and a final frame of the finished sign is captured when
the job completes.

Recording starts when the artwork does, so a video opens on a blank sign
rather than on the brush-calibration strokes the job wipes off first.

The tab lists every recording newest first with its frame count and size, and
shows what the current job is recording. Select one and:

- **Watch** plays it back in a window inside RustPainter. Space plays and
  pauses, the arrow keys step one frame, and the slider scrubs. Double-clicking
  a recording in the list opens it too.
- **Save video** writes the whole recording to one file, in the container
  picked beside it.
- **Speed** is one slider for both playback and the exported frame rate, so
  what you watched is what you save. It reads out the frame rate and how much
  of the paint job one second of video covers at it, which is the number worth
  choosing by.

The buttons at the right of that row open the timelapse folder, open the
selected recording's folder, look for recordings again, and delete the
selected recordings; each names itself on hover.

Recordings are selected the way files are anywhere else: click one, shift-click
another for the whole run between them, ctrl-click to add or drop one, and
Ctrl+A for all of them. Watching, saving, and opening a folder work on one
recording at a time and stay disabled while several are picked; deleting does
not, so a run of old recordings is cleared out with one confirmation and one
press of Delete. The line under the list totals up what a multiple selection
holds before it goes. A recording the current paint job is still writing to is
left out of the batch rather than failing it.

AVI (Motion JPEG) and animated GIF are written by RustPainter itself and need
nothing installed. MP4 is offered as well when `ffmpeg` is on `PATH` (or named
by the `RUST_PAINTER_FFMPEG` environment variable); it is smaller and travels
better. Exporting never touches the captured PNG frames, so a recording can be
saved again at a different speed or in a different format, and a cancelled or
failed export deletes its half-written file rather than leaving a video that
stops mid-paint.

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

## Run reports

Every paint job writes a diagnostic folder under `runs/` in the app's data
directory, whether it finishes, is aborted, or fails. A sign that came out
wrong is usually explained hours later, and a folder of canvas frames alone
cannot answer the questions that decide the diagnosis - so the job records
them while it runs:

- **`plan.png`** - the exact image the plan reproduces. Replanning it from the
  source afterwards repeats the pipeline but not the fonts, the palette, or
  the settings of that day, and a reference that is off by a few rows is worse
  than none.
- **`run.json`** - the plan's structure (size, stroke count, how many of those
  are single cells, and every color group **in painting order**), the settings
  and profile exactly as they were at the countdown, and the brush the job
  measured for itself stated against the cells it had to fit:
  `smallestBrushCells` above 1 means every stroke repainted its neighbours,
  and `planRowsOverSignRows` above 1 means the plan asked for more rows than
  the sign's texture can store.
- **`screen_start.png`** - one full-desktop capture as the artwork begins. It
  is the only record of what the game was set to: brush shape and opacity, the
  Size field, the selected color, and how the sign was framed.
- **`progress.csv`** - a trace sampled every couple of seconds carrying the
  timelapse frame count, so any frame showing something wrong maps back to the
  color and stroke being painted when it was taken. Pauses are traced too,
  since an unexplained gap is the first thing to look for when a run took
  longer than it should have.
- **`canvas_final.png`** - the sign as the run left it.

The screen capture and the final canvas are written on background threads, so
recording never stalls a stroke. A report that cannot be written is logged and
skipped: a missing diagnostic is a nuisance and a lost paint job is hours.

## Local data and logs

Profiles, settings, calibration reference captures, timelapse frames, run reports, and logs are stored under `%LOCALAPPDATA%\RustPainter` when available. Deleting that folder resets the application; exporting or copying its JSON files is enough to back up calibration data.

## Known limitations

- The app cannot know Rust's internal brush radius, native sign resolution, or exact picker gradient up front. Brush measurement infers the first two from probe strokes; the picker gradient still needs calibration and small test strokes.
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
    text_render.py     one text renderer for the editor and the baked plan
  calibration.py       full-screen rectangle selector
  profiles.py          profile JSON persistence
  settings.py          defaults and local settings
  image_processing.py  composition, background removal, palette reduction
  color_mapping.py     RGB/HSV picker mapping
  brush_calibration.py probe-stroke measurement of Rust's Size numbers
  color_calibration.py painted-chart response fitting
  coordinates.py       logical/screen coordinate conversion
  paint_plan.py        horizontal-run planning and estimates
  paint_optimizer.py   artist-style optimized planning (modes, brushes)
  input_controller.py  SendInput and dry-run input
  hotkeys.py           Windows global hotkeys
  painter.py           resumable, abortable execution
  screen.py            DPI/display/focus/capture helpers
  run_report.py        per-run diagnostic bundle for after the fact
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
