"""Do clicks at the color box's corners register, or only interior clicks?

Clicks the saturation/value box at exact corners and at 2%-inset points,
photographing the picker after each, so the cursor's landing spot settles
whether Rust accepts edge clicks at all.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.input_controller import create_system_input_controller  # noqa: E402
from app.models import ScreenRect  # noqa: E402
from app.profiles import ProfileStore  # noqa: E402
from app.screen import capture_region  # noqa: E402
from app.picker_calibration import trim_to_widget  # noqa: E402
from tools._safety import Guard  # noqa: E402
from tools.decimal_probe import _data_directory, _focus_rust  # noqa: E402


def main() -> None:
    output = Path("diagnostic/corner")
    output.mkdir(parents=True, exist_ok=True)
    store = ProfileStore(_data_directory() / "profiles")
    profile = store.get_default() or store.list_profiles()[0]
    raw_box = ScreenRect(
        profile.color_box.left, profile.color_box.top,
        profile.color_box.width, profile.color_box.height,
    )
    box = trim_to_widget(capture_region(raw_box), raw_box)
    print(f"trimmed box: {box.left},{box.top} {box.width}x{box.height}")
    park = (box.left - 60, box.top + box.height // 2)

    _focus_rust()
    guard = Guard(create_system_input_controller(), budget_seconds=60)

    cases = [
        ("corner_bottom_right", box.left + box.width - 1, box.top + box.height - 1),
        ("inset_bottom_right", box.left + round(box.width * 0.98),
         box.top + round(box.height * 0.98)),
        ("corner_top_right", box.left + box.width - 1, box.top),
        ("inset_top_right", box.left + round(box.width * 0.98),
         box.top + round(box.height * 0.02)),
        ("center", box.left + box.width // 2, box.top + box.height // 2),
    ]
    shots = []
    for name, x, y in cases:
        guard.check()
        guard.input.click(x, y, hold_seconds=0.09)
        guard.commanded(x, y)
        time.sleep(0.3)
        guard.park(park, settle=0.35)
        shot = capture_region(box).convert("RGB")
        shot.save(output / f"{name}.png")
        shots.append((name, shot))
        print(f"  clicked {name} at ({x},{y})")

    w = max(s.width for _, s in shots) + 260
    h = sum(s.height + 6 for _, s in shots)
    sheet = Image.new("RGB", (w, h), (28, 28, 28))
    draw = ImageDraw.Draw(sheet)
    offset = 0
    for name, shot in shots:
        draw.text((6, offset + 20), name, fill=(230, 230, 230))
        sheet.paste(shot, (260, offset))
        offset += shot.height + 6
    sheet.save(output / "corner_probe.png")
    print("wrote", output / "corner_probe.png")


if __name__ == "__main__":
    main()
