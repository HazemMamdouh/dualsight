"""DualSight visual identity: graphite black / silver / muted gold, HUD-style.

One module owns every colour and font so the live dashboard, the rendered videos, the
matplotlib report and the HTML page all read as one system.

Category hues are a validated categorical set on the dark surface (lightness band, chroma
floor, all-pairs CVD and normal-vision separation pass); identity is always repeated in text
labels, never colour-only.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Tuple

# ---------------------------------------------------------------- surfaces & ink
BLACK = "#0A0B0D"        # page / canvas (graphite black)
PANEL = "#111317"        # chart surface
PANEL_2 = "#171A1F"      # raised card
GRID = "#22252B"
EDGE = "#343841"
SILVER = "#CDD3DB"       # primary ink
SILVER_2 = "#8E95A0"     # secondary ink
SILVER_3 = "#5C626C"     # muted ink
WHITE = "#F1F3F6"

# ---------------------------------------------------------------- accent (muted gold - brand marks only)
ACCENT = "#C9A54E"
ACCENT_BRIGHT = "#E8C76F"   # live markers, active state
ACCENT_DEEP = "#3A3120"     # fills, low end of ramps

# ---------------------------------------------------------------- data colours
# categorical: validated on the dark surface (lightness band, chroma, all-pairs CVD and
# normal-vision separation pass); labels always repeat the category in text
CATEGORY_HEX = {
    "person": "#D95926",          # burnt orange
    "vehicle": "#3987E5",         # blue
    "animal": "#199E70",          # aqua
    "vessel": "#8F7AD9",
    "aircraft": "#C9A54E",
    "carried-object": "#D55181",
    "unknown": "#7D838C",
}
CHANNEL_HEX = {"thermal": "#C9A54E", "rgb": "#A9B8C9", "fused": WHITE}   # gold = heat, silver = visible
GATE_HEX = {"defog": "#8C7438", "deblur": "#4D6680"}

# sequential ramp for heat (single hue: graphite -> gold -> pale)
HEAT_RAMP = ["#0A0B0D", "#1E1B14", "#3A3120", "#6B5728", "#A88A3E", "#D9BC6A", "#F6E7BE"]

# line weights (thin, precise)
LW = 1.2            # matplotlib data lines (pt)
LW_HUD = 1          # OpenCV data lines (px)

# backwards-compatible aliases
RED, RED_BRIGHT, RED_DEEP, RED_GLOW = ACCENT, ACCENT_BRIGHT, ACCENT_DEEP, ACCENT
SURFACE = PANEL
TEXT = SILVER
TEXT_2 = SILVER_2

FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "fonts")
FONTS = {
    "display": os.path.join(FONT_DIR, "Orbitron-VF.ttf"),
    "ui": os.path.join(FONT_DIR, "Rajdhani-SemiBold.ttf"),
    "ui_bold": os.path.join(FONT_DIR, "Rajdhani-Bold.ttf"),
    "ui_medium": os.path.join(FONT_DIR, "Rajdhani-Medium.ttf"),
    "mono": os.path.join(FONT_DIR, "JetBrainsMono-VF.ttf"),
}


def hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def hex_to_bgr(h: str) -> Tuple[int, int, int]:
    r, g, b = hex_to_rgb(h)
    return (b, g, r)


def category_hex(cat: str) -> str:
    return CATEGORY_HEX.get(cat, CATEGORY_HEX["unknown"])


def category_bgr(cat: str) -> Tuple[int, int, int]:
    return hex_to_bgr(category_hex(cat))


@lru_cache(maxsize=64)
def pil_font(kind: str, size: int):
    """Truetype font for PIL; falls back to PIL's default if the bundled file is missing."""
    from PIL import ImageFont

    path = FONTS.get(kind, FONTS["ui"])
    try:
        f = ImageFont.truetype(path, size)
        if kind in ("display", "mono"):
            try:  # variable fonts: pick a weight
                f.set_variation_by_name("Bold" if kind == "display" else "Medium")
            except Exception:
                pass
        return f
    except Exception:
        return ImageFont.load_default()


def register_mpl_fonts() -> None:
    """Make the bundled fonts available to matplotlib."""
    try:
        from matplotlib import font_manager

        for p in FONTS.values():
            if os.path.exists(p):
                font_manager.fontManager.addfont(p)
    except Exception:
        pass
