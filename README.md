# RustPainter

<p align="center">
  <img src="RustPainterIcon.png" alt="RustPainter icon" width="128">
  &nbsp;&nbsp;
  <a href="https://discord.gg/VMNQtgD3v"><img src="assets/discord.svg" alt="Join the RustPainter Discord" width="128"></a>
</p>

this takes an image and paints it onto a Rust sign for you. It uses normal mouse input, so you set up the parts of Rust's paint UI it needs to click first.

## Download and run

You need Windows 10 or 11, plus Rust running in borderless or windowed mode.

1. Download `RustPainter.exe` from the [latest release](https://github.com/YeheyaMohammad01/RustPainter/releases/latest).
2. Open the downloaded file. There is no installer or Python setup needed.
3. Open Rust, then open the sign's paint screen and leave it there while you set RustPainter up.

If Windows shows a warning for the downloaded `.exe`, open its Properties and check **Unblock** if that option is there. The app is unsigned, so Windows may be a little weird about it.

## Use it

1. In RustPainter, calibrate the **canvas**, **color box**, and **hue bar** by dragging just inside each matching area in Rust's paint UI.
2. Drag an image into RustPainter, or click **Choose image**.
3. Check the preview and start with a small, low-color image so you can make sure your calibration is right.
4. Focus Rust during the countdown, then start painting.

`F8` starts, pauses, or resumes a paint job. `F10` stops it immediately and releases the mouse button.

keep a hand near F10 on your first few runs and dont leave it painting unattended.

## Need help?

[<img src="assets/discord.svg" alt="Discord" width="28" valign="middle"> Join the Discord](https://discord.gg/VMNQtgD3v)

## Run from source

If you want to run the project yourself, install 64-bit Python 3.11 through 3.14, then run:

```powershell
git clone https://github.com/YeheyaMohammad01/RustPainter.git
cd RustPainter
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```
