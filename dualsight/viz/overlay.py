"""Live operator dashboard - HUD styling (graphite black / silver / muted gold).

Layout
    header      : product mark, live/replay status, stream clock
    two panels  : CH1 thermal (LWIR) | CH2 visible, with motion masks, tracked targets,
                  trails, headings, and gate status plates
    telemetry   : fog-score gauge (with the gate threshold) | camera-shift gauge
    strip       : rolling motion activity per channel, gate lanes, event ticks, situation cards

Geometry is drawn with OpenCV; all text goes through one PIL pass per frame so the bundled
fonts (Orbitron, Rajdhani, JetBrains Mono - SIL OFL) render crisply.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ..motion.tracker import project_box
from . import style as S

BG = S.hex_to_bgr(S.BLACK)
PANEL = S.hex_to_bgr(S.PANEL)
PANEL2 = S.hex_to_bgr(S.PANEL_2)
GRID = S.hex_to_bgr(S.GRID)
EDGE = S.hex_to_bgr(S.EDGE)
SILVER = S.hex_to_bgr(S.SILVER)
SILVER2 = S.hex_to_bgr(S.SILVER_2)
SILVER3 = S.hex_to_bgr(S.SILVER_3)
ACC = S.hex_to_bgr(S.ACCENT)
ACCB = S.hex_to_bgr(S.ACCENT_BRIGHT)
ACCD = S.hex_to_bgr(S.ACCENT_DEEP)
TH = S.hex_to_bgr(S.CHANNEL_HEX["thermal"])
VI = S.hex_to_bgr(S.CHANNEL_HEX["rgb"])
G_FOG = S.hex_to_bgr(S.GATE_HEX["defog"])
G_BLUR = S.hex_to_bgr(S.GATE_HEX["deblur"])


@dataclass
class TrackView:
    id: int
    box: List[float]                 # reference coordinates
    label: str = "unknown"
    fine: str = ""
    conf: float = 0.0
    sources: Sequence[str] = ()
    heading: Optional[float] = None
    trail: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class FrameState:
    frame: int
    t: float
    fog_on: bool
    fog_score: float
    deblur_on: bool
    cam_shift: float
    motion_thermal: float
    motion_rgb: float
    tracks: List[TrackView]
    mask_thermal: Optional[np.ndarray] = None   # processing-resolution masks (optional)
    mask_rgb: Optional[np.ndarray] = None
    thermal_method: str = ""
    n_events: Dict[str, int] = field(default_factory=dict)
    cam_dx: float = 0.0                          # measured rig motion (ref px / frame)
    cam_dy: float = 0.0
    proc_fps: float = 0.0                        # live processing rate (0 in offline renders)
    transition: bool = False                     # atmospheric-transition guard active
    mode: str = ""                               # LIVE | REPLAY (default: LIVE when proc_fps > 0)


class _Text:
    """Deferred text items rendered in a single PIL pass."""

    def __init__(self) -> None:
        self.items: List[tuple] = []

    def add(self, xy, text, kind="ui", size=14, color=S.SILVER, anchor="la"):
        self.items.append((xy, text, kind, size, color, anchor))

    def width(self, text, kind="ui", size=14) -> int:
        f = S.pil_font(kind, size)
        return int(f.getlength(text))

    def render(self, canvas: np.ndarray) -> np.ndarray:
        if not self.items:
            return canvas
        img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img)
        for xy, text, kind, size, color, anchor in self.items:
            d.text(xy, text, font=S.pil_font(kind, size), fill=S.hex_to_rgb(color) if isinstance(color, str) else color,
                   anchor=anchor)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def _alpha_rect(img, p1, p2, color, alpha):
    x1, y1 = max(0, p1[0]), max(0, p1[1])
    x2, y2 = min(img.shape[1], p2[0]), min(img.shape[0], p2[1])
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    roi[:] = (roi * (1 - alpha) + np.array(color, np.float32) * alpha).astype(np.uint8)


def _brackets(img, x1, y1, x2, y2, color, L=14, t=2):
    for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * L, cy), color, t, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * L), color, t, cv2.LINE_AA)


def _tint(panel: np.ndarray, mask: Optional[np.ndarray], color=ACCB, alpha=0.22) -> None:
    if mask is None or not mask.any():
        return
    m = cv2.resize(mask, (panel.shape[1], panel.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    over = panel.copy()
    over[m] = color
    cv2.addWeighted(over, alpha, panel, 1 - alpha, 0, panel)
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(panel, cnts, -1, color, 1, cv2.LINE_AA)


def _compass(deg: Optional[float]) -> str:
    if deg is None:
        return "--"
    return ["E", "SE", "S", "SW", "W", "NW", "N", "NE"][int(((deg + 22.5) % 360) // 45)]


class Dashboard:
    def __init__(self, cfg: dict, ref_size: Tuple[int, int], thermal_size: Tuple[int, int],
                 H_th2ref: np.ndarray, fps: float, gate_cfg: Optional[dict] = None) -> None:
        self.pw = int(cfg["panel_width"])
        rw, rh = ref_size
        self.ph = int(round(self.pw * rh / rw))
        self.ref_size = ref_size
        self.thermal_size = thermal_size
        self.ref_to_rgb_panel = np.diag([self.pw / rw, self.ph / rh, 1.0])
        tw, th = thermal_size
        self.ref_to_th_panel = np.diag([self.pw / tw, self.ph / th, 1.0]) @ np.linalg.inv(H_th2ref)
        self.cmap = cfg.get("thermal_cmap", "gray")
        self.show_masks = bool(cfg.get("show_masks", True))
        self.window_s = float(cfg["timeline_s"])
        self.fps = fps
        self.hist: Deque[tuple] = deque(maxlen=int(self.window_s * fps * 1.2) + 4)
        self.event_marks: Deque[Tuple[float, str]] = deque(maxlen=64)
        self._seen_ids: set = set()
        g = gate_cfg or {}
        self.fog_on_thr = float(g.get("fog_on", 0.40))
        self.cam_thr = float(g.get("cam_shift", 2.5))
        self.gut = 10
        self.header_h = 46
        self.tele_h = 34
        self.strip_h = 168
        self.W = 2 * self.pw + 3 * self.gut
        self.H = self.header_h + self.ph + self.tele_h + self.strip_h + self.gut
        # static scanline / vignette texture for the camera panels
        yy, xx = np.mgrid[0:self.ph, 0:self.pw].astype(np.float32)
        vign = 1.0 - 0.28 * (((xx / self.pw - 0.5) ** 2 + (yy / self.ph - 0.5) ** 2) / 0.5)
        scan = np.where((np.arange(self.ph) % 3) == 0, 0.92, 1.0).astype(np.float32)[:, None]
        self.fx = (vign * scan)[..., None]

    # ---------------------------------------------------------------- panels
    def _thermal_panel(self, gray: np.ndarray) -> np.ndarray:
        g = cv2.resize(gray, (self.pw, self.ph), interpolation=cv2.INTER_LINEAR)
        if self.cmap == "inferno":
            return cv2.applyColorMap(g, cv2.COLORMAP_INFERNO)
        if self.cmap == "turbo":
            return cv2.applyColorMap(g, cv2.COLORMAP_TURBO)
        if self.cmap in ("theme", "goldhot"):  # theme LUT: graphite -> gold -> white
            lut = np.zeros((256, 1, 3), np.uint8)
            stops = [(0, S.BLACK), (90, S.ACCENT_DEEP), (170, S.ACCENT), (230, "#F2E4BE"), (255, S.WHITE)]
            for (a, ca), (b, cb) in zip(stops[:-1], stops[1:]):
                A, B = np.array(S.hex_to_bgr(ca), float), np.array(S.hex_to_bgr(cb), float)
                for i in range(a, b + 1):
                    lut[i, 0] = (A + (B - A) * (i - a) / max(1, b - a)).astype(np.uint8)
            return cv2.LUT(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), lut)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    def _draw_track(self, panel: np.ndarray, tv: TrackView, to_panel: np.ndarray, txt: _Text, ox: int, oy: int) -> None:
        b = project_box(tv.box, to_panel)
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        col_hex = S.category_hex(tv.label)
        col = S.hex_to_bgr(col_hex)
        # fading trail
        if len(tv.trail) > 1:
            tr = np.float32(tv.trail)
            if len(tr) >= 5:  # running median: suppress single-frame jumps of the blob centre
                tr = np.float32([np.median(tr[max(0, i - 2): i + 3], axis=0) for i in range(len(tr))])
            pts = np.float32([[p[0], p[1], 1.0] for p in tr]).T
            q = to_panel @ pts
            q = (q[:2] / q[2:3]).T.astype(np.int32)
            n = len(q)
            for i in range(1, n):
                a = i / n
                c = tuple(int(cc * (0.15 + 0.85 * a)) for cc in col)
                cv2.line(panel, tuple(q[i - 1]), tuple(q[i]), c, 1, cv2.LINE_AA)
        _alpha_rect(panel, (x1, y1), (x2, y2), col, 0.08)
        cv2.rectangle(panel, (x1, y1), (x2, y2), tuple(int(c * 0.55) for c in col), 1, cv2.LINE_AA)
        _brackets(panel, x1, y1, x2, y2, col, L=max(6, min(16, (x2 - x1) // 3)), t=2)
        if tv.heading is not None:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            r = max(14, (x2 - x1) // 2 + 6)
            a = np.deg2rad(tv.heading)
            cv2.arrowedLine(panel, (cx, cy), (int(cx + r * np.cos(a)), int(cy + r * np.sin(a))), col, 1, cv2.LINE_AA, tipLength=0.3)
        # label plate (2 lines)
        name = (tv.fine or tv.label).upper()
        l1 = f"#{tv.id}  {name}"
        src = "T+V" if len(tv.sources) > 1 else ("T" if "thermal" in tv.sources else "V")
        l2 = f"{tv.label.upper()}  {tv.conf:.2f}  {src}  {_compass(tv.heading)}" if tv.conf > 0 else f"{tv.label.upper()}  {src}  {_compass(tv.heading)}"
        w = max(txt.width(l1, "ui_bold", 15), txt.width(l2, "mono", 10)) + 16
        h = 34
        px, py = x1, y1 - h - 4
        if py < 0:
            py = y2 + 4
        px = min(max(0, px), panel.shape[1] - w - 1)
        _alpha_rect(panel, (px, py), (px + w, py + h), PANEL, 0.82)
        cv2.rectangle(panel, (px, py), (px + 3, py + h), col, -1)
        cv2.line(panel, (px, py + h), (px + w, py + h), col, 1, cv2.LINE_AA)
        txt.add((ox + px + 9, oy + py + 2), l1, "ui_bold", 15, S.WHITE)
        txt.add((ox + px + 9, oy + py + 20), l2, "mono", 10, col_hex)

    def _frame_panel(self, canvas, x, y, txt: _Text, tag: str, title: str, chips: List[Tuple[str, str, bool]]):
        cv2.rectangle(canvas, (x - 1, y - 1), (x + self.pw, y + self.ph), EDGE, 1)
        _brackets(canvas, x - 4, y - 4, x + self.pw + 3, y + self.ph + 3, ACC, L=16, t=1)
        # title plate
        _alpha_rect(canvas, (x, y), (x + self.pw, y + 26), BG, 0.72)
        txt.add((x + 8, y + 4), tag, "display", 11, S.ACCENT)
        txt.add((x + 8 + txt.width(tag, "display", 11) + 8, y + 3), title, "ui_bold", 15, S.SILVER)
        cx = x + self.pw - 6
        for text, color, active in reversed(chips):
            w = txt.width(text, "mono", 10) + 14
            cx -= w
            if active:
                cv2.rectangle(canvas, (cx, y + 5), (cx + w - 4, y + 21), S.hex_to_bgr(color), -1)
                txt.add((cx + 5, y + 7), text, "mono", 10, S.WHITE)
            else:
                cv2.rectangle(canvas, (cx, y + 5), (cx + w - 4, y + 21), SILVER3, 1)
                txt.add((cx + 5, y + 7), text, "mono", 10, S.SILVER_2)
            cx -= 4

    def _gauge(self, canvas, x, y, w, value, vmax, thr, label, unit, color, txt: _Text):
        txt.add((x, y + 2), label, "display", 9, S.SILVER_2)
        gx = x + 118
        gw = w - 118 - 92
        n = 30
        seg = gw / n
        filled = int(round(min(1.0, value / vmax) * n))
        for i in range(n):
            sx = int(gx + i * seg)
            c = color if i < filled else GRID
            cv2.rectangle(canvas, (sx, y + 4), (int(sx + seg) - 2, y + 16), c, -1)
        tx = int(gx + min(1.0, thr / vmax) * gw)
        cv2.line(canvas, (tx, y), (tx, y + 20), SILVER, 1)
        txt.add((tx + 3, y - 8), "GATE", "mono", 8, S.SILVER_2)
        txt.add((gx + gw + 10, y), f"{value:5.2f} {unit}", "mono", 12, S.WHITE if value >= thr else S.SILVER_2)

    # ---------------------------------------------------------------- compose
    def compose(self, thermal_gray: np.ndarray, rgb: np.ndarray, st: FrameState) -> np.ndarray:
        txt = _Text()
        W, H = self.W, self.H
        canvas = np.full((H, W, 3), BG, np.uint8)
        # faint tech grid on the background
        for gx in range(0, W, 32):
            cv2.line(canvas, (gx, 0), (gx, H), (15, 14, 13), 1)
        for gy in range(0, H, 32):
            cv2.line(canvas, (0, gy), (W, gy), (15, 14, 13), 1)

        tp = self._thermal_panel(thermal_gray)
        rp = cv2.resize(rgb, (self.pw, self.ph), interpolation=cv2.INTER_AREA)
        tp = (tp * self.fx).astype(np.uint8)
        rp = (rp * self.fx).astype(np.uint8)
        if self.show_masks:
            _tint(tp, st.mask_thermal)
            _tint(rp, st.mask_rgb)
        x_t, x_r, y_p = self.gut, 2 * self.gut + self.pw, self.header_h
        for tv in st.tracks:
            self._draw_track(tp, tv, self.ref_to_th_panel, txt, x_t, y_p)
            self._draw_track(rp, tv, self.ref_to_rgb_panel, txt, x_r, y_p)
            if tv.id not in self._seen_ids:
                self._seen_ids.add(tv.id)
                self.event_marks.append((st.t, tv.label))
        canvas[y_p:y_p + self.ph, x_t:x_t + self.pw] = tp
        canvas[y_p:y_p + self.ph, x_r:x_r + self.pw] = rp

        # header
        cv2.line(canvas, (0, self.header_h - 8), (W, self.header_h - 8), EDGE, 1)
        cv2.rectangle(canvas, (self.gut, 12), (self.gut + 4, 32), ACC, -1)
        txt.add((self.gut + 12, 9), "DUALSIGHT", "display", 20, S.WHITE)
        txt.add((self.gut + 12 + txt.width("DUALSIGHT", "display", 20) + 14, 15),
                "ALL-WEATHER  THERMAL + VISIBLE  MOTION INTELLIGENCE", "ui_medium", 15, S.SILVER_2)
        clock = f"T+{int(st.t // 60):02d}:{st.t % 60:05.2f}   F{st.frame:05d}"
        cw = txt.width(clock, "mono", 13)
        txt.add((W - self.gut - cw, 15), clock, "mono", 13, S.SILVER)
        live = (st.mode or ("LIVE" if st.proc_fps > 0 else "REPLAY")).upper()
        if st.proc_fps > 0:
            fps_s = f"{st.proc_fps:4.1f} FPS"
            txt.add((W - self.gut - cw - 40 - txt.width(live, "display", 11) - txt.width(fps_s, "mono", 11) - 18, 16),
                    fps_s, "mono", 11, S.SILVER_2)
        lx = W - self.gut - cw - 26 - txt.width(live, "display", 11)
        if live != "LIVE" or int(st.t * 2) % 2 == 0:
            cv2.circle(canvas, (lx, 22), 5, ACCB, -1, cv2.LINE_AA)
        txt.add((lx + 11, 15), live, "display", 11, S.ACCENT)

        th_chips = ([("FOG TRANSITION", S.ACCENT_DEEP, True)] if st.transition else []) + [
                    (f"FOG {st.fog_score:.2f}", S.PANEL_2, False),
                    (f"DEFOG {st.thermal_method.upper()}" if st.fog_on else "DEFOG STANDBY", S.GATE_HEX["defog"], st.fog_on)]
        rgb_chips = [(f"CAM {st.cam_shift:4.1f}PX/F", S.PANEL_2, False),
                     ("DEBLUR ACTIVE" if st.deblur_on else "DEBLUR STANDBY", "#55606F", st.deblur_on)]
        self._frame_panel(canvas, x_t, y_p, txt, "CH1", "THERMAL  LWIR", th_chips)
        self._frame_panel(canvas, x_r, y_p, txt, "CH2", "VISIBLE  RGB", rgb_chips)

        # telemetry gauges
        ty = y_p + self.ph + 10
        self._gauge(canvas, x_t, ty, self.pw, st.fog_score, 1.0, self.fog_on_thr, "FOG INDEX", "", G_FOG, txt)
        self._gauge(canvas, x_r, ty, self.pw, st.cam_shift, max(10.0, self.cam_thr * 6), self.cam_thr, "CAMERA MOTION", "px/f", G_BLUR, txt)
        # rig motion vector on the RGB panel (bottom-right compass)
        if st.cam_shift > 0.05:
            cx, cy, r = x_r + self.pw - 34, y_p + self.ph - 34, 22
            cv2.circle(canvas, (cx, cy), r, EDGE, 1, cv2.LINE_AA)
            cv2.line(canvas, (cx - r, cy), (cx + r, cy), GRID, 1)
            cv2.line(canvas, (cx, cy - r), (cx, cy + r), GRID, 1)
            m = max(1e-6, st.cam_shift)
            k = min(1.0, m / max(10.0, self.cam_thr * 6)) * r
            cv2.arrowedLine(canvas, (cx, cy), (int(cx + st.cam_dx / m * k), int(cy + st.cam_dy / m * k)),
                            ACCB if st.deblur_on else SILVER, 2, cv2.LINE_AA, tipLength=0.35)
            txt.add((cx, cy + r + 2), "RIG", "mono", 8, S.SILVER_3, "ma")

        self.hist.append((st.t, st.motion_thermal, st.motion_rgb, st.fog_on, st.deblur_on, len(st.tracks)))
        self._strip(canvas, y_p + self.ph + self.tele_h, W, st, txt)
        return txt.render(canvas)

    def _strip(self, canvas: np.ndarray, y0: int, W: int, st: FrameState, txt: _Text) -> None:
        g = self.gut
        h = self.strip_h
        side_w = 300
        x0, x1 = g + 46, W - g - side_w - 16
        top, bot = y0 + 62, y0 + h - 26
        cv2.rectangle(canvas, (g, y0 + 4), (W - g, y0 + h), PANEL, -1)
        cv2.rectangle(canvas, (g, y0 + 4), (W - g, y0 + h), EDGE, 1)
        _brackets(canvas, g, y0 + 4, W - g, y0 + h, ACC, L=12, t=1)
        txt.add((g + 10, y0 + 10), "MOTION ACTIVITY", "display", 10, S.SILVER)
        txt.add((g + 10 + txt.width("MOTION ACTIVITY", "display", 10) + 10, y0 + 9), f"LAST {int(self.window_s)} S  ·  % OF FRAME MOVING",
                "ui_medium", 13, S.SILVER_2)
        t_now, t_min = st.t, st.t - self.window_s
        hist = [hh for hh in self.hist if hh[0] >= t_min]
        X = lambda t: int(x0 + (t - t_min) / self.window_s * (x1 - x0))
        for k in range(5):
            yy = int(bot - k * (bot - top) / 4)
            cv2.line(canvas, (x0, yy), (x1, yy), GRID, 1)
        for s in range(0, int(self.window_s) + 1, 5):
            xx = X(t_now - s)
            cv2.line(canvas, (xx, bot), (xx, bot + 4), SILVER3, 1)
            txt.add((xx, bot + 6), f"-{s}s" if s else "NOW", "mono", 9, S.SILVER_3 if s else S.ACCENT, "ma")
        if hist:
            ymax = max(0.01, max(max(hh[1], hh[2]) for hh in hist) * 1.2)
            Y = lambda v: int(bot - v / ymax * (bot - top))
            # gate lanes
            for k, (lane, col, name) in enumerate(((3, G_FOG, "DEFOG"), (4, G_BLUR, "DEBLUR"))):
                yy = y0 + 34 + k * 9
                txt.add((x0 - 8, yy - 6), name, "mono", 8, S.SILVER_3, "ra")
                cv2.line(canvas, (x0, yy), (x1, yy), GRID, 3)
                for a, b in zip(hist[:-1], hist[1:]):
                    if a[lane]:
                        cv2.line(canvas, (X(a[0]), yy), (X(b[0]), yy), col, 3)
            for idx, col in ((2, VI), (1, TH)):
                pts = np.int32([[X(hh[0]), Y(hh[idx])] for hh in hist])
                poly = np.vstack([pts, [[pts[-1][0], bot], [pts[0][0], bot]]])
                over = canvas.copy()
                cv2.fillPoly(over, [poly], col)
                cv2.addWeighted(over, 0.10, canvas, 0.90, 0, canvas)
                cv2.polylines(canvas, [pts], False, col, S.LW_HUD, cv2.LINE_AA)
            txt.add((x0 - 6, top - 6), f"{ymax * 100:.1f}%", "mono", 9, S.SILVER_3, "ra")
            txt.add((x0 - 6, bot - 6), "0", "mono", 9, S.SILVER_3, "ra")
        for (te, cat) in self.event_marks:
            if te >= t_min:
                xe = X(te)
                c = S.category_bgr(cat)
                cv2.line(canvas, (xe, top), (xe, bot), tuple(int(v * 0.5) for v in c), 1)
                cv2.fillPoly(canvas, [np.int32([[xe - 5, bot + 1], [xe + 5, bot + 1], [xe, bot - 7]])], c)
        cv2.line(canvas, (x1, top - 4), (x1, bot), ACCB, 1)
        # legend
        lx = x0 + 6
        for name, col in (("THERMAL", S.CHANNEL_HEX["thermal"]), ("VISIBLE", S.CHANNEL_HEX["rgb"])):
            cv2.line(canvas, (lx, top + 8), (lx + 16, top + 8), S.hex_to_bgr(col), 2)
            txt.add((lx + 22, top + 2), name, "mono", 9, S.SILVER_2)
            lx += 90

        # situation cards
        sx = W - g - side_w
        txt.add((sx, y0 + 10), "SITUATION", "display", 10, S.SILVER)
        ev = st.n_events or {}
        total = sum(ev.values())
        act = max(st.motion_thermal, st.motion_rgb) * 100
        cards = [("EVENTS", f"{total:02d}", S.WHITE), ("TRACKS", f"{len(st.tracks):02d}", S.ACCENT),
                 ("ACTIVITY", f"{act:4.1f}%", S.SILVER)]
        cw = (side_w - 16) // 3
        for i, (k, v, c) in enumerate(cards):
            cx = sx + i * (cw + 8)
            cv2.rectangle(canvas, (cx, y0 + 30), (cx + cw, y0 + 76), PANEL2, -1)
            cv2.line(canvas, (cx, y0 + 30), (cx + cw, y0 + 30), ACC if i == 1 else EDGE, 2)
            txt.add((cx + 8, y0 + 34), k, "mono", 9, S.SILVER_3)
            txt.add((cx + 8, y0 + 47), v, "display", 17, c)
        yy = y0 + 86
        cats = sorted(ev.items(), key=lambda kv: -kv[1]) or [("none", 0)]
        for i, (k, v) in enumerate(cats[:4]):
            cx = sx + (i % 2) * (side_w // 2)
            cy = yy + (i // 2) * 24
            if k != "none":
                cv2.rectangle(canvas, (cx, cy + 4), (cx + 10, cy + 14), S.category_bgr(k), -1)
            txt.add((cx + 16, cy), f"{k.upper()}", "ui_bold", 14, S.SILVER)
            txt.add((cx + side_w // 2 - 22, cy + 1), f"{v}", "mono", 12, S.WHITE, "ra")
