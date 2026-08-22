# RustPainter

RustPainter is a local desktop utility for Windows 10/11 that recreates an image in Rust's sign-painting UI by using ordinary screen coordinates and synthesized mouse input. It does **not** read game memory, inject code, hook graphics, modify game files, or attempt to bypass anti-cheat.

The application keeps all profiles and settings on your PC. Every coordinate used for painting comes from a rectangle that you calibrate on your own display.

## Install

**You need:** Windows 10/11, and Rust running in borderless or
windowed mode (not exclusive fullscreen).

### Download the app (easiest)

Grab **`RustPainter.exe`** from the
[latest release](https://github.com/YeheyaMohammad01/RustPainter/releases/latest)
and double-click it. That single file is the whole application - there is no
installer, no Python to set up, and no admin rights needed. To uninstall,
delete the file.

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

**Hotkeys:** `F8` start/resume, `F9` pause, `F10` stop. Stop immediately
releases any held mouse button.

Keep a hand near **F10** during your first runs, and do not leave the tool
painting unattended. Everything below is detail you only need when tuning.

## Features

- Dark, rust-textured workspace built around the image -> calibrate -> paint flow, with advanced controls tucked into Settings
- Loads PNG, JPEG, WebP, BMP, TIFF and other Pillow-supported images by clicking either preview, browsing, or dropping a file anywhere in the window
- Fit, fill/crop, and stretch composition with a live paint simulation, with the Fill crop draggable directly on the source image so the sign keeps the part of the picture you meant
- Adjustable painting resolution, palette size, dithering, sharpening, transparency, and fit background
- One-click background removal that leaves the backdrop unpainted, by detected or picked color, with a smart matcher for gradients, vignettes and photographic backdrops that also clears the halo off a cut-out subject
- Multiple draggable text layers edited right on the Source tab - inline editing, resize handles, Ctrl+D or Ctrl+C to copy, Delete to remove, and a bracketed outline showing the part of the image the sign will hold - while the Rust preview shows the text baked in exactly as it will paint, marks itself read-only, and offers a way back to the Source tab if you try to edit there
- Text editing conveniences you would expect from a graphics app: select several layers with a rubber band, Shift+click, Ctrl+click or Ctrl+A and restyle or drag them together, snap to the middle and edges of the sign and to the other layers while dragging (Alt bypasses), align and spread buttons, and arrow-key nudging
- Per-layer gradients and outlines, both drawn by the same renderer the paint plan bakes, so the letters on the Source tab are the letters the sign receives
- Undo and redo over the text layers alone (Ctrl+Z / Ctrl+Y anywhere in the window, mid-typing included), so a whole drag or a run of keystrokes steps back as one edit
- Text sized as a fraction of the canvas, so a caption keeps its proportions when the quality preset changes the painting resolution
- Named profiles per sign/UI layout, each inheriting the current calibration
- Drag calibration for the canvas, color box, hue bar, and - for automatic brush sizing - the numeric Size field and Rust's clear button, with an on-screen overlay to verify them
- An anti-AFK break (off by default): every so often the job saves the sign, jumps, reopens the sign, and carries on, so a server that kicks idle players sees one moving
- Brush sizing measured from the sign itself at the start of every job: a few probe strokes fit what Rust's Size numbers really cover, the sign is wiped clean again, and only then does the artwork go down
- Optimization modes (Exact / Quality / Balanced / Fast) that plan like a painter: perceptually identical colors merge, insignificant specks are absorbed, and large areas are filled with the largest safe brush before details go on top, with the preview showing exactly what will be painted
- Overpaint stroke merging that typically removes 10-40% of strokes without changing the finished image
- Speed presets (Relaxed / Standard / Fast / Turbo) over fully adjustable timing, with 1 ms Windows timer resolution while painting - every hold and settle floored at a frame of the game's paint UI, and long drags paced by the sign's own texel pitch, so the fastest preset is also an accurate one
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
9. Begin with a small, low-color test image. Keep the stop hotkey available.

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

### Viewing the Rust preview

The plan has one color per logical cell, and the preview tab has to scale
those cells up to fill itself. The **Smooth** switch beside the heading picks
how: on, cells blend into each other the way Rust filters the sign's texture
when it draws it, which is the closer guess at the finished sign; off, every
cell is a hard square, which exaggerates the grid but shows exactly what each
stroke will paint. The choice is remembered between sessions.

### Sharpening for the sign

The same filtering happens in game, and it is why an image shrunk to the sign
looks softer there than it did on screen: the sign has a few hundred texels a
side, and the game blends each one into its neighbours when it draws them.
**Sharpen**, in Quick settings beside the preview, puts back about the edge contrast that
blending takes away, before the image is painted. **Light** (the default)
restores roughly what the filter costs a line and suits nearly everything;
**Strong** is for line art that should bite, at the price of a faint light
halo beside dark lines; **Off** paints the plain downscale. Only images that
were actually reduced are sharpened - an enlarged image has lost nothing and
sharpening its blocks would only ring - and a subject cut free of its
background is sharpened against itself rather than against the backdrop that
was removed. Sharpening cannot add detail the texels do not have; the only cure
for that is a larger sign.

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
Region cleanup knows what an edge looks like. A downscaled line is nothing
but tiny regions of in-between color along its sides, so a speck is only
absorbed into a neighbor when its color is not the blend between that
neighbor and a contrasting one, and it goes to the neighbor closest in color
rather than the one it touches most - which lets a hair strand's short
segments unify into one strand instead of dissolving into the fill around
them. Dithered images keep their deliberate speckle: region cleanup turns
itself off when dithering is enabled.

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

Default global hotkeys are **F8** start/resume, **F9** pause, and **F10** stop. Stop is designed to release any held mouse button immediately.

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

Exclusive fullscreen applications can prevent overlays, screenshots, focus checks, or synthetic input from behaving normally. Borderless fullscreen is recommended. If Rust or Windows is running at a different privilege level than RustPainter, Windows may reject input; run both at the same privilege level.

## Settings that require experimentation

Rust brush behavior can vary with sign type, selected in-game brush, frame rate, and UI scale. These controls are grouped under **Settings → Painting** so the main workspace stays focused:

- Painting speed preset (Relaxed / Standard / Fast / Turbo); editing any timing value switches it to Custom
- Stroke merging (Off / Balanced / Maximum) under Paint Quality - only used by
  Exact paint mode; the other modes merge through the optimizer and the box
  says so
- Automatic brush sizing, which types the Size number a logical cell needs (optional; needs the Size value box and the clear button calibrated)
- Logical pixel spacing
- Stroke duration/speed and interpolation step
- Mouse-down time for dots
- Picker and inter-stroke delays
- Canvas inset and logical resolution

Start with a low-resolution 8-color test. If adjacent rows bleed together, reduce the in-game brush size, increase logical spacing, or lower the logical resolution. The speed presets differ in how much margin they leave above the game's frame floors (see *Speed and accuracy* below), not in whether the paint lands; if strokes still have gaps, raise the touch-up passes before slowing anything.

**Stroke merging** exploits painting order: colors are painted from most to least frequent, so an earlier color may paint across pixels that a later color repaints anyway. The final image is identical, but fragmented regions (text backgrounds, dithered gradients) need far fewer mouse strokes. *Balanced* merges across gaps of up to 6 logical pixels and is the default; *Maximum* produces the fewest strokes but can spend extra time traveling across very long overpainted spans. *Off* exists for comparing against exact strokes and is never faster. The paint-plan panel reports how many strokes merging removed. Only **Exact** paint mode reads this setting - Quality, Balanced and Fast merge automatically as part of the optimizer, and the box shows *Automatic* while one of them is selected.

### Speed and accuracy

The speed preset is not one knob but a bundle of timings, and they fail in
different ways. Rust samples its paint UI at about 15 FPS, so anything it is
asked to notice - a press, a picker click, the color a click just changed -
has to last until the next frame samples it. That is a cliff, not a slope:
below about 70 ms the event is simply not seen (an overnight run once lost
708 cells to a short hold), and above it extra waiting buys nothing. The
press hold, the settles after the hue, S/V and Size-field clicks, and the
pause before the picker between colors therefore have **floors** of one
frame that no preset goes under; the Advanced timing spinboxes stop there,
and a value from an older profile below a floor is run at the floor and
logged once at the start of the job. The gap between strokes is different -
the press and release are events the game orders itself - so its floor is
only a few milliseconds of slack.

Overshoot is the other failure, and it belongs to the stroke speed alone: a
long drag at thousands of pixels a second with a coarse interpolation step is
sampled mid-flight, and paints past its ends and skips texels in the middle.
A dab or a run of a few texels cannot overshoot at any speed, because the
frame hold at its far end is what lands it. So the painter paces each drag
by its length and by the sign's measured texel pitch (the grid probe's, or
the brush measurement's, or the logical cell's): short runs go at the set
speed, and a long drag is capped at 250 texels per second with a cursor
event on every texel, whatever the preset says. Relaxed on a 320x240 canvas
runs at about 130 texels/s and was always clean; the old Turbo ran at about
730 and was not.

Which makes **Turbo** simply "the floors everywhere": as fast as the game can
take input with nothing dropped, and Relaxed the same with margin on top. The
touch-up pass stays the backstop for what the game drops anyway; what this
prevents up front is overshoot, which paints wrong texels the touch-up would
then have to repaint.

### How long it will take

A stroke costs the same whether it paints one cell or thirty. Every press is
held for at least a frame of Rust's 15 FPS paint UI (see above), and at any
usable speed setting nearly every stroke is shorter than that hold, so a run
is essentially *strokes × about 85 ms* plus a color change per color. The
estimate in the plan panel is priced from exactly the rules the painter
executes with - the frame hold per stroke, the held picker clicks per color
change, the retyped Size field per brush change, the countdown and the brush
measurement before the first stroke - and the one machine-dependent term, the
overhead of input calls and timer slack per stroke, is learned from every run
and kept in `timing.json` next to the settings. On the two runs used to check
it the estimate landed within 4% of the clock. Long drags are priced at the
texel-paced rate they are actually driven at. The speed preset barely moves
the estimate, because it barely moves the run: the way to paint faster is
fewer strokes - fewer colors, a lower resolution, or an optimizing paint
mode.

While painting, the time left is the same prediction corrected by the pace
measured so far, with progress advancing in predicted seconds rather than in
cells - so the bar no longer races through the big, long-stroke colors and
then crawls through the small ones. A touch-up pass is timed against its own
clock, and its status line says how long it is expected to take before it
starts.

Automatic brush sizing works with a solid square or circle. Spray/noise
brushes do not have a stable footprint and must be sized manually.

Because the brush is measured on the sign rather than inferred from Rust's
preview tile, an impossible plan is refused before the first stroke instead of
discovered as smearing. If the smallest brush Rust offers is wider than one
logical cell, RustPainter says so and names the resolution that would fit -
that limit is the sign's own texture resolution, which no setting can raise.

That ceiling is also why the quality presets stop making a difference on a
small sign. A 320x240 sign holds fewer cells than High or Very High ask for,
so both used to be held at 320x240 and paint exactly what Max paints - a
setting that looks finer and is not. Once a job has measured the sign, the
presets it cannot deliver are greyed out in the list instead, each carrying
the reason as a tooltip, and a selection sitting on one moves to Max, which
is the honest name for the same size. Quick settings says why underneath,
and the plan summary repeats the size next to the stroke counts. Max and
Custom are never greyed - Custom can still ask for less - and on a bigger
sign the presets separate again and nothing is greyed at all.

A one-cell brush targets the full logical cell plus half a sign texel: the
sign renders strokes snapped to its own texture rows, and a brush sized
exactly to the row pitch still lands half a texel narrow wherever snapping
rounds down - which shows as bare stripes across the painting. The half-texel
overlap is invisible instead: boundaries are texel-quantized either way, and
the later-painted color owns the shared texel. Perfectly edge-to-edge pixels
still favor the **square** brush: a circle of one cell across leaves gaps at
the cell corners. Along a horizontal run the two shapes behave alike; the
difference shows at run ends and on isolated pixels.

The probe strokes measure both axes of the brush at once: the band's height
is its vertical footprint, and the amount its ends stick out past the drag is
its horizontal one. The two only agree when the calibrated rectangle has
exactly the sign texture's aspect ratio. (A live probe of the large wooden
sign measured 4.08px per Size unit across against 4.53px down - a 320x240
texture under a 1.20 rectangle - which is why assuming a square footprint
left a bare stripe beside every column.) Rust's brush is square in the sign's
own texels, so the Size number is chosen to match the *rows* exactly, and on
cells wider than tall the remaining width is covered by geometry instead:
each stroke drags a little further sideways, a dab becoming a tiny horizontal
drag - the trick a sign painter uses when the roller is narrower than the
board. Within the probed range, converting between Size numbers and painted
fractions interpolates between the recorded probes rather than reading the
fitted line: the line can miss a nearby probe by a few pixels, and a few
pixels is the entire seam budget of a one-cell brush.

### Finding the sign's texel grid

Strokes are laid out on the sign texture's own grid, not on the hand-dragged
rectangle. The rectangle covers the sign only to hand-drag precision, so
cells laid out on it are a fraction of a texel off the texture's grid, and
because stamps land on whole texels that fraction accumulates until a later
neighbour's stamp eats a texel of the cell before it - visible as unevenly
wide cells in a sign texture downloaded from the game.

The grid is measured, not inferred, at the start of every job, right after
the brush. Paint lands in texture space: wherever the cursor is inside a
texel, the stamp covers that whole texel. So the job stamps the smallest
brush along a row one screen pixel at a time and watches the stamp stay put,
stay put, then jump a whole texel - that staircase is the grid showing
through, and it gives the texel pitch to a few percent and the cursor
positions that cross from one texel to the next. A ladder of stamps further
and further along the sign then tightens the pitch: each rung is far enough
out to be counted exactly with the pitch known so far, and locating it pins
the pitch down further, until the far side of the sign is a whole number of
texels with nothing to round. Each rung is stamped more than once on
separate rows and read by majority, so a dab the game drops a texel astray
cannot miscount it. Then dabs aimed at the texels the rectangle's edges fall
in, and their neighbours, show which texel is the first and which the last -
the sign's extent, observed rather than inferred. Both axes; a little over a
hundred dabs, all wiped with the brush probes.

Two things a live sign taught the probe. The game draws a frame over the
outer edge of the texture, so the visible quad is not the texture's extent:
the last column had 1.5 px in view and 2.6 px under the frame, painted but
invisible in the UI - and on the sign in the world. A texel the visible edge
cuts through therefore counts even when no stamp can be seen on it. And the
game takes paint clicks only on the texture, frame or no frame, so the mouse
is held on whole pixels inside the measured texture, with the rectangle as a
one-pixel-slack outer bound.

The second thing is that *where the cursor has to be* is not the same grid as
*where the texels are drawn*. The canvas is drawn flat, but the cursor is
mapped as if onto the sign in the world: its lattice has a slightly
different pitch from the rendered one, is sheared - the column boundaries
sat three pixels further left at the bottom of the sign than at the top -
and keystoned a little on top of that. One offset cannot describe that; on
a 320-texel sign it put a sixth of a test lattice's dots a texel over in one
corner. So the short staircases are repeated in bands across each axis, and
the cursor map is fitted as a plane with a twist from every boundary they
bracket. Painting maps every cell through it; verification reads every cell
back from the rendered lattice. Painted one cell per texel on a measured
grid, the brush is typed as exactly Size 1 - the half-texel overlap the brush
model adds exists to bridge a grid off by a fraction of a texel, and on an
exact grid it only spills into the neighbour.

Measured on a large artist canvas (320x240): a lattice of 1,200 single-texel
dots landed 1,200 of 1,200 on the intended texel, 0.17 x 0.12 px rms from its
centre, worst 0.4 px. The brush-derived inference the app used before put
165 of them nowhere and 459 of the rest a texel or more off.

What comes out is the texture's size in texels (no table of sign sizes is
consulted, so a sign of a size no table lists is counted just the same),
exactly where each texel is drawn, and exactly where to click for each.
Max quality plans one cell per counted texel. The measurement is absolute
screen pixels, so a job that can measure never reuses a stored one: it
describes where the sign sat on screen the day it was taken.

With **Automatic brush sizing** off the probe cannot run - it types the
smallest brush and needs the sign wiped afterwards - so the job paints on the
grid the profile's last measurement stored, as long as that grid still sits
on the calibrated rectangle; without one it falls back to the rectangle,
whose hand-dragged edges are good to about half a texel. Live, that half
texel was the difference between a clean sign and one with every other row
bare through the middle: the rectangle's row pitch was 0.0077 px longer than
the texture's, which walked the aim onto a texel boundary by row 60 and
across it by row 110, so alternate rows landed on their neighbours. Leave
sizing on for native-resolution work, or at least run one sizing-on job on
the sign first so there is a grid to reuse. The same job also captures the
sign before its first stroke and, if that capture is one bare surface rather
than an earlier picture, keeps it as the touch-up pass's bare reference -
otherwise a dropped row reads as "some other color" and, past a few hundred
such cells, is set aside as a capture that cannot resolve cells.

Only a measurement that snaps - stamps within one texel agreeing on where they
landed, every ladder rung a whole number of texels out, the grid sitting on
the calibrated rectangle - is used. Anything else is logged and the job falls
back to the older inference: the brush measurement's texel size, snapped to
the nearest known texture size and anchored at the rectangle's corner, with
the brush probes' measured rendering bias aimed out. For the resolution
ceiling the brush count is read against Rust's own sign-size table
(`tools/sign_sizes.json`): a Size unit is about 0.8 of a texel, so the raw
count runs a quarter high, but corrected and paired with the rectangle's
shape it names the sign's exact entry - the XXL artist canvas reads ~649
units and is 1024x512 - where the plain snap had called it 640 rows and
Max planned a quarter more cells than the sign can show. Typing a table
edge into a Custom resolution derives the other edge from the table too, so
1024 on a hand-dragged 1.997:1 rectangle gives 512 rows, not 513. ``painting.measure_texel_grid``
in settings.json turns the probe off outright if it ever needs to be.

Probe colours are chosen against what is already on the sign where each
batch lands: a probe in the colour of an earlier probe at the same place
reads as no change at all, which is how a second measurement on an uncleared
sign once found its scout stroke and scout stamp "did not change the sign".

Colors that share a boundary still meet in a one-texel blend line - the
brush's own edge falloff, visible on a checkerboard and invisible on ordinary
artwork - which is the game's rendering, not a placement error.

For development there is also `tools/dump_sign_sizes.py`, which reads the
texture sizes Rust's own prefabs declare out of the game's asset bundles
(files on disk, nothing running) with UnityPy. The bundles embed their type
trees, so no game assembly is needed, and they open lazily, so even the
multi-gigabyte ones scan in under a minute. Its last output is committed as
`tools/sign_sizes.json`: 26 distinct sizes across the deployable signs,
frames, canvases and banners, and they are not what the community tables
say - the wooden signs are 256x128 / 512x256 / 512x256 / 1024x256 (small,
medium, large, huge), the picture frames include 205x256 and 256x192, the
artist canvases run 192x256, 320x240, 256x640, 512x512 and 1024x512, and the
DLC frames 128x175, 320x240, 320x256, 128x320. That table is what the
fallback snapping uses; the probe itself never consults it.

### Strokes the game never sees, and the touch-up that repairs them

Rust samples the mouse a frame at a time, and its paint UI has been measured
running at about 15 FPS. A press and release that both fall inside one frame
can be sampled as nothing, and the painter has no way of knowing: it pushes
synthetic input into Windows and never hears back from the game. Dabs were
protected against this early on by being held for most of a frame. Reading a
finished five-hour sign back showed the same thing happening to *short drags*:
at a fine painting resolution a run of a few cells is only a few screen
pixels, over in under ten milliseconds, and every hole in that sign was such a
run missing from the middle of a stroke. Every stroke's press now lasts at
least as long as a dab's hold, with a short drag keeping the button down at
its far end until it has; long strokes already spend frames moving and are not
slowed at all.

What the game dropped anyway is caught by the **touch-up** passes under
**Settings → Painting**. After the artwork is down, the sign is captured and
each cell compared with the plan, and the cells that came out wrong are
repainted. The comparison is relative, so the sign's lighting and material
never count against it, and it measures each color against *what that color
actually looks like on this sign*, read off the capture itself - two palette
entries the sign renders identically can therefore never be mistaken for one
another, which is what used to send a quarter of a 256-color sign back for
repainting. A cell is repainted when it reads as bare sign (the job captures
the freshly cleared sign for exactly this reference, and any area the plan
leaves unpainted serves as well), when it looks like nothing the plan painted,
or when it decisively took another color. The default is two passes: the
touch-up strokes are short and can be dropped by the game exactly as the
originals were, and the second capture is what catches that.

On a plan finer than the game can paint - cells narrower than Rust's smallest
brush, or under two screen pixels across - the touch-up fills holes only. A
cell that reads as the wrong color there is as likely a neighbour's paint as a
mistake, and "correcting" it with a brush wider than the cell would smear the
neighbours it was read from.

Wrong-color verdicts are also held to the shapes painting actually fails in.
A picker click that misses paints a *whole color* wrong, and a color read as
mostly wrong is repainted whole. Wrong cells sprinkled a few per color through
colors that are otherwise right are something else: nothing in the painting
loop miscolors one cell in five at random, so once they cover more than 5% of
the sign (and at least 500 cells) they are taken as the capture failing to
resolve cells and are left alone, with a warning in the log. Before this rule
a 512-wide sign read at two screen pixels per cell sent 21% of its cells back
for "recoloring" - 35,000 strokes and most of an hour to repaint a sign that
was already right.

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
2. Under **Settings → Color**, click **Prepare Calibration Chart**. This replaces the imported image and selects Stretch, Very Fast, 32 colors, no dithering, and no sharpening.
3. Paint the complete chart through the normal Start/countdown workflow. Any older correction is automatically bypassed for this chart.
4. Leave the finished chart visible in Rust and click **Measure Painted Chart**.
5. Focus Rust during the capture countdown. The app samples 32 large swatches, rejects inconsistent captures, and saves the measured correction to that profile.
6. Reset/use a fresh sign and reload the artwork. Correction is applied automatically to future paint jobs.

Once measured, the Rust preview renders artwork through the model as well, so the preview and the sign agree. Colors the material can reach look unchanged; colors outside its measured gamut show as the muted version Rust will actually produce instead of a promise it cannot keep.

The chart deliberately consumes paint on one test sign. Re-measure after changing sign material, display/graphics color behavior, or the main canvas/picker calibration. **Clear Color Correction** restores direct RGB-to-picker mapping.

## Safety behavior

- Starting uses a visible countdown so you can focus Rust.
- With the foreground guard enabled, every populated selector must match: the configured window-title fragment and, when supplied, the executable name. The expected process defaults to `RustClient.exe`. Loss of focus pauses and releases the mouse.
- F9 pauses at the next short cancellation checkpoint. While a job is paused the speed preset, the advanced timing, touch-up passes, and the safety guards stay editable; the job resumes on the new values. Everything that shaped the job (image, plan, calibration, brush sizing) stays locked until it finishes.
- F10 aborts, clears pending work, and releases the mouse.
- **Anti-AFK** (off by default, Settings > Safety): every N minutes (adjustable, 30 by default) the job clicks Rust's **Save changes** button to leave the painting UI, presses Space to jump, waits a second, clicks to open the sign again, and continues from the same stroke - re-selecting its color and brush size first. It relies on your character still facing the sign, which it is if you were looking at it when the job started and have not touched the mouse since. Turning it on makes the **Save button** calibration needed (drag just inside Rust's Save changes button), and Start says so until it is set. Closing the UI with Save keeps everything painted so far.
- **UI guard** (on by default, Settings > Safety): once a second the job looks at the calibrated **colour box**, **hue bar**, and - when calibrated - the **Clear** and **Save** buttons, and pauses when they are no longer on the screen: a server restart, a kick, a disconnect, or the sign closed by hand, none of which the other guards notice because the Rust window is still in front and the cursor still goes where it is sent. Each widget is fingerprinted as the job starts (the colour box on saturation and value only, since its hue follows the colour being painted) and recognised by structure, so a highlight or a tint does not count and a dark wall where a dark button was does not pass. The UI counts as gone when more of the widgets are missing than present, and only after two looks half a second apart. The job also refuses to start painting unless the hue bar is where it was calibrated, which catches a countdown that ran out before the sign was open. Open the sign again and resume to continue from the same stroke; the anti-AFK break closes the sign on purpose and is exempt, but if the sign has not reopened a few seconds after its E press the job pauses instead of painting into the game world.
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
  paint_plan.py        horizontal-run planning
  paint_timing.py      what a plan costs in seconds, learned per machine
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
