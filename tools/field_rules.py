"""Type edge-case values into the Size field and photograph what it keeps.

No painting - just typing and reading back, so it answers granularity and
range questions in a few seconds: does 1.35 hold, or only halves?  What do
0.99 and 150 clamp to?
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.input_controller import create_system_input_controller  # noqa: E402
from app.models import ScreenRect  # noqa: E402
from app.profiles import ProfileStore  # noqa: E402
from app.screen import capture_region  # noqa: E402
from tools._safety import Guard  # noqa: E402
from tools.decimal_probe import _data_directory, _focus_rust  # noqa: E402


VK_OEM_PERIOD = 0xBE
CLICK_HOLD = 0.09

VALUES = ("1.35", "1.05", "1.75", "2.33", "0.99", "150")


def run(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    store = ProfileStore(_data_directory() / "profiles")
    profile = store.get_default() or store.list_profiles()[0]
    box = ScreenRect(
        profile.brush_size_box.left,
        profile.brush_size_box.top,
        profile.brush_size_box.width,
        profile.brush_size_box.height,
    )
    park = (
        int(profile.color_box.left + profile.color_box.width / 2),
        int(profile.color_box.top + profile.color_box.height / 2),
    )

    _focus_rust()
    guard = Guard(create_system_input_controller(), budget_seconds=60)
    shots = []
    for text in VALUES:
        guard.check()
        guard.input.click(
            box.left + box.width / 2, box.top + box.height / 2, hold_seconds=CLICK_HOLD
        )
        guard.commanded(box.left + box.width / 2, box.top + box.height / 2)
        time.sleep(0.15)
        for key in ("BACKSPACE",) * 6 + ("DELETE",) * 6:
            guard.input.press_key(key, hold_seconds=0.03)
            time.sleep(0.02)
        for char in text:
            guard.input.press_key(
                VK_OEM_PERIOD if char == "." else char, hold_seconds=0.03
            )
            time.sleep(0.02)
        guard.input.press_key("ENTER", hold_seconds=0.03)
        time.sleep(0.25)
        guard.park(park, settle=0.3)
        shot = capture_region(box).convert("RGB")
        shots.append((text, shot))
        print(f"  typed {text}", flush=True)

    scale = 3
    w = max(s.width for _, s in shots) * scale + 160
    h = sum(s.height * scale + 4 for _, s in shots)
    sheet = Image.new("RGB", (w, h), (30, 30, 30))
    from PIL import ImageDraw

    draw = ImageDraw.Draw(sheet)
    offset = 0
    for text, shot in shots:
        draw.text((6, offset + 20), f"typed {text}", fill=(220, 220, 220))
        sheet.paste(
            shot.resize((shot.width * scale, shot.height * scale), Image.LANCZOS),
            (160, offset),
        )
        offset += shot.height * scale + 4
    sheet.save(output / "field_rules.png")
    print(f"Wrote {output / 'field_rules.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    run(parser.parse_args().out)


if __name__ == "__main__":
    main()
