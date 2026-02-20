"""Offline analytics report: *when* and *where* motion happened and *what* caused it.

Reads a run directory produced by ``dualsight run`` and writes ``report/``:

* ``timeline.png``     - motion activity per channel, fog score vs. gate, gate lanes, event Gantt
* ``heatmap_rgb.png`` / ``heatmap_thermal.png`` - event heat per channel over the median background
* ``trajectories.png`` - every event's path, coloured by category, with heading arrows
* ``sensors.png``      - which sensor saw each event (thermal only / both / visible only)
* ``zones.png``        - motion-seconds per zone of a configurable grid
* ``categories.png``   - events and motion-time per category
* ``headings.png``     - heading rose (image-plane direction of travel)
* ``performance.png``  - per-stage latency
* ``index.html``       - self-contained report (images and fonts embedded) with the event log
* ``events.csv``       - flat event table for spreadsheets / SIEM ingestion

Visual identity: graphite black / silver / muted gold (see ``viz.style``).
"""
from __future__ import annotations

import base64
import csv
import html
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from ..viz import style as S  # noqa: E402

S.register_mpl_fonts()
HEAT = LinearSegmentedColormap.from_list("dualsight_heat", S.HEAT_RAMP)
FOG_HEX = S.GATE_HEX["defog"]
BLUR_HEX = S.GATE_HEX["deblur"]
BOTH_HEX = "#6E7A8A"

plt.rcParams.update({
    "figure.facecolor": S.BLACK, "axes.facecolor": S.PANEL, "savefig.facecolor": S.BLACK,
    "axes.edgecolor": S.EDGE, "axes.labelcolor": S.SILVER_2, "xtick.color": S.SILVER_2, "ytick.color": S.SILVER_2,
    "text.color": S.SILVER, "axes.grid": True, "grid.color": S.GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
    "font.family": ["Rajdhani", "DejaVu Sans"], "font.weight": 600, "axes.titlesize": 12, "axes.titlelocation": "left",
    "axes.titlecolor": S.SILVER, "legend.frameon": False, "axes.axisbelow": True,
    "legend.labelcolor": S.SILVER, "xtick.labelsize": 10, "ytick.labelsize": 10,
})
TITLE_FONT = {"family": ["Orbitron", "DejaVu Sans"], "size": 11, "color": S.SILVER}
import logging  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)



def _title(ax, title: str, sub: str = "") -> None:
    ax.set_title(title.upper(), fontdict=TITLE_FONT, loc="left", pad=22 if sub else 8)
    if sub:
        ax.annotate(sub, xy=(0, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points",
                    fontsize=10.5, color=S.SILVER_2, va="bottom", ha="left")


def _accent(fig) -> None:
    """Thin red rule + corner ticks framing every figure."""
    fig.add_artist(plt.Line2D([0.0, 0.06], [0.997, 0.997], transform=fig.transFigure, color=S.ACCENT, lw=1.5))
    fig.add_artist(plt.Line2D([0.06, 1.0], [0.997, 0.997], transform=fig.transFigure, color=S.EDGE, lw=0.6))


def _save(fig, path: str) -> None:
    _accent(fig)
    fig.savefig(path, dpi=140, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def _load(run: str):
    import pandas as pd

    frames = pd.read_csv(os.path.join(run, "frames.csv"))
    with open(os.path.join(run, "events.json")) as f:
        events = json.load(f)
    with open(os.path.join(run, "run.json")) as f:
        meta = json.load(f)
    return frames, events, meta


def _fmt_t(s: float) -> str:
    return f"{int(s // 60):02d}:{s % 60:04.1f}"


def _spans(t: np.ndarray, flag: np.ndarray):
    out, start = [], None
    for ti, f in zip(t, flag):
        if f and start is None:
            start = ti
        elif not f and start is not None:
            out.append((start, ti))
            start = None
    if start is not None:
        out.append((start, t[-1]))
    return out


def _roots(run: str) -> Dict[int, int]:
    mp = os.path.join(run, "track_merges.json")
    merges = {int(k): int(v) for k, v in json.load(open(mp)).items()} if os.path.exists(mp) else {}

    def root(t):
        while t in merges:
            t = merges[t]
        return t

    return root


# ======================================================================================
# WHEN
# ======================================================================================

def chart_timeline(frames, events, path: str, gate_cfg: dict) -> None:
    t = frames["t"].to_numpy() - frames["t"].iloc[0]
    n_ev = max(1, len(events))
    fig = plt.figure(figsize=(13, 8.6))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.3, 1.0, 0.5, 0.36 * n_ev + 0.6], hspace=0.55)

    ax = fig.add_subplot(gs[0])
    for ch, name in (("rgb", "visible"), ("thermal", "thermal")):
        y = frames[f"motion_{ch}"].to_numpy() * 100
        c = S.CHANNEL_HEX[ch]
        ax.fill_between(t, 0, y, color=c, alpha=0.08, lw=0)
        ax.plot(t, y, color=c, lw=S.LW, label=f"{name} channel")
    ax.set_ylabel("moving pixels (%)")
    _title(ax, "When · motion activity", "share of each frame flagged as moving, per sensor")
    ax.legend(loc="upper right", ncol=2)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(bottom=0)

    ax2 = fig.add_subplot(gs[1], sharex=ax)
    fs = frames["fog_score"].to_numpy()
    ax2.fill_between(t, 0, fs, color=S.ACCENT_DEEP, alpha=0.6, lw=0)
    ax2.plot(t, fs, color=S.ACCENT, lw=S.LW)
    on, off = float(gate_cfg.get("on_score", 0.4)), float(gate_cfg.get("off_score", 0.3))
    ax2.axhline(on, color=S.SILVER_2, lw=0.7, ls=(0, (4, 3)))
    ax2.axhline(off, color=S.SILVER_3, lw=0.7, ls=(0, (2, 3)))
    ax2.text(t[-1], on, f" ON {on:.2f}", color=S.SILVER, fontsize=9.5, va="bottom", ha="left")
    ax2.text(t[-1], off, f" OFF {off:.2f}", color=S.SILVER_3, fontsize=9.5, va="top", ha="left")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("fog score")
    _title(ax2, "Thermal fog index", "classical check (dynamic range + edge energy) with hysteresis thresholds")

    ax3 = fig.add_subplot(gs[2], sharex=ax)
    for k, (col, hexc) in enumerate((("fog_on", FOG_HEX), ("deblur_on", BLUR_HEX))):
        ax3.add_patch(FancyBboxPatch((t[0], 1 - k - 0.26), t[-1] - t[0], 0.52, boxstyle="round,pad=0,rounding_size=0.2",
                                     fc=S.GRID, ec="none", mutation_aspect=0.05))
        for a, b in _spans(t, frames[col].to_numpy() > 0):
            ax3.add_patch(FancyBboxPatch((a, 1 - k - 0.26), max(b - a, 1e-3), 0.52, boxstyle="round,pad=0,rounding_size=0.2",
                                         fc=hexc, ec="none", mutation_aspect=0.05))
    ax3.set_yticks([1, 0])
    ax3.set_yticklabels(["DEFOG", "DEBLUR"], fontfamily="JetBrains Mono", fontsize=9)
    ax3.set_ylim(-0.6, 1.6)
    ax3.grid(False)
    ax3.spines["left"].set_visible(False)
    ax3.tick_params(axis="y", length=0)
    _title(ax3, "Enhancement gates")

    ax4 = fig.add_subplot(gs[3], sharex=ax)
    for k, e in enumerate(events):
        c = S.category_hex(e["category"])
        ax4.add_patch(FancyBboxPatch((e["start_s"], k - 0.3), e["end_s"] - e["start_s"], 0.6,
                                     boxstyle="round,pad=0,rounding_size=0.15", fc=c, ec="none", alpha=0.95,
                                     mutation_aspect=0.2))
        src = "T+V" if len(e["seen_by"]) > 1 else ("T" if "thermal" in e["seen_by"] else "V")
        ax4.text(e["end_s"] + 0.35, k, f"#{e['id']}  {(e['class'] or e['category']).upper()}   {e['confidence']:.2f} · {src}",
                 va="center", fontsize=10, color=S.SILVER)
    ax4.set_yticks([])
    ax4.set_ylim(len(events) - 0.4, -0.7)
    ax4.set_xlabel("time (s)")
    ax4.grid(axis="y", visible=False)
    _title(ax4, "Motion events", "bar = event lifetime · colour = detected category · T/V = thermal / visible")
    cats = sorted({e["category"] for e in events})
    if cats:
        handles = [plt.Rectangle((0, 0), 1, 1, color=S.category_hex(c)) for c in cats]
        ax4.legend(handles, [c.upper() for c in cats], loc="lower right", bbox_to_anchor=(1.0, 1.0),
                   ncol=len(cats), fontsize=9.5)
    for a in (ax, ax2, ax3):
        plt.setp(a.get_xticklabels(), visible=False)
    _save(fig, path)


# ======================================================================================
# WHERE
# ======================================================================================

def _heat_overlay(bg: np.ndarray, heat: np.ndarray) -> np.ndarray:
    h, w = bg.shape[:2]
    hm = cv2.resize(heat, (w, h), interpolation=cv2.INTER_LINEAR)
    hm = cv2.GaussianBlur(hm, (0, 0), max(1.5, w / 300))
    if hm.max() > 0:
        hm = hm / hm.max()
    hm = np.clip(hm, 0, 1) ** 0.75
    rgba = HEAT(hm)
    base = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    base = 0.38 * (0.4 * base + 0.6 * base.mean(axis=2, keepdims=True))  # darkened, desaturated backdrop
    a = np.where(hm > 0.05, 0.25 + 0.7 * hm, 0.0)[..., None]
    return np.clip(base * (1 - a) + rgba[..., :3] * a, 0, 1)


def event_heat(run: str, events: List[dict], ref_size, channel: str, down: int = 4) -> np.ndarray:
    """Footprint heat of the final events, counting only frames in which ``channel`` saw the object.

    The raw per-pixel motion accumulators (``heatmap_<channel>.npy``) stay in the run directory, but
    they also contain motion that was rejected as noise, so the report maps the *events*."""
    rw, rh = ref_size
    heat = np.zeros((rh // down, rw // down), np.float32)
    tmp = np.zeros_like(heat, np.uint8)
    ids = {e["id"] for e in events}
    root = _roots(run)
    with open(os.path.join(run, "frames.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            for tr in r["tracks"]:
                if root(tr["id"]) not in ids or channel not in tr["src"]:
                    continue
                x1, y1, x2, y2 = [v / down for v in tr["box"]]
                tmp[:] = 0
                cv2.ellipse(tmp, (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                            (max(1, int((x2 - x1) / 2)), max(1, int((y2 - y1) / 2))), 0, 0, 360, 1, -1)
                heat += tmp
    return heat


def chart_heatmap(run: str, ch: str, title: str, path: str, ref_size, heat: np.ndarray, H_th2ref=None) -> bool:
    bgp = os.path.join(run, f"background_{'rgb' if ch == 'rgb' else 'thermal'}.png")
    if not os.path.exists(bgp) or heat.max() <= 0:
        return False
    bg = cv2.imread(bgp)
    if bg.ndim == 2:
        bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    if ch == "thermal" and H_th2ref is not None:
        rw, rh = ref_size
        hh, hw = heat.shape
        Sm = np.diag([hw / rw, hh / rh, 1.0])
        th_h, th_w = bg.shape[:2]
        heat_to_thermal = np.linalg.inv(np.array(H_th2ref, np.float64)) @ np.linalg.inv(Sm)
        heat = cv2.warpPerspective(heat.astype(np.float32), heat_to_thermal, (th_w, th_h))
    img = _heat_overlay(bg, heat)
    fig, ax = plt.subplots(figsize=(8.5, 8.5 * img.shape[0] / img.shape[1] + 0.9))
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(S.EDGE)
    _title(ax, title, "event footprints accumulated over time on the median background")
    sm = plt.cm.ScalarMappable(cmap=HEAT)
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.012)
    cb.set_ticks([0.02, 0.98])
    cb.set_ticklabels(["RARE", "FREQUENT"], fontfamily="JetBrains Mono", fontsize=8.5)
    cb.outline.set_edgecolor(S.EDGE)
    _save(fig, path)
    return True


def _smooth_path(p: np.ndarray, k: int = 5) -> np.ndarray:
    """Running median of the box-centre path: removes single-frame jumps (blob splits/merges)."""
    if len(p) < k:
        return p
    h = k // 2
    out = np.array([np.median(p[max(0, i - h): i + h + 1], axis=0) for i in range(len(p))])
    return out


def chart_trajectories(run: str, events: List[dict], ref_size, path: str) -> None:
    bgp = os.path.join(run, "background_rgb.png")
    rw, rh = ref_size
    fig, ax = plt.subplots(figsize=(11, 11 * rh / rw + 0.9))
    if os.path.exists(bgp):
        bg = cv2.cvtColor(cv2.imread(bgp), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        bg = 0.32 * (0.4 * bg + 0.6 * bg.mean(axis=2, keepdims=True))
        ax.imshow(bg, extent=(0, rw, rh, 0))
    placed: list = []
    for e in events:
        p = _smooth_path(np.array([q[:2] for q in e["path"]], np.float32))
        if len(p) < 2:
            continue
        c = S.category_hex(e["category"])
        ax.plot(p[:, 0], p[:, 1], color=c, lw=1.4, solid_capstyle="round")
        ax.scatter(p[0, 0], p[0, 1], s=46, facecolor=S.BLACK, edgecolor=c, lw=1.4, zorder=3)
        ax.annotate("", xy=p[-1], xytext=p[max(0, len(p) - 4)],
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.4, mutation_scale=14))
        mid = p[len(p) // 2]
        ly = mid[1] - 26
        while any(abs(q[0] - mid[0]) < 0.22 * rw and abs(q[1] - ly) < 0.045 * rh for q in placed):
            ly -= 0.05 * rh
        placed.append((mid[0], ly))
        ax.text(mid[0], ly, f"#{e['id']} {(e['class'] or e['category']).upper()}  {_fmt_t(e['start_s'])}–{_fmt_t(e['end_s'])}",
                fontsize=10, color=S.WHITE, ha="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc=S.PANEL, ec=c, lw=0.8, alpha=0.92))
    ax.set_xlim(0, rw)
    ax.set_ylim(rh, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(S.EDGE)
    _title(ax, "Where · event trajectories", "ring = first seen · arrow = last seen · in RGB reference coordinates")
    cats = sorted({e["category"] for e in events})
    if cats:
        ax.legend([plt.Line2D([0], [0], color=S.category_hex(c), lw=2) for c in cats],
                  [c.upper() for c in cats], loc="lower right", ncol=len(cats))
    _save(fig, path)


def chart_sensors(run: str, events: List[dict], path: str) -> None:
    """Per event: frames seen by thermal only / both / visible only."""
    root = _roots(run)
    ids = {e["id"] for e in events}
    cnt: Dict[int, Counter] = defaultdict(Counter)
    with open(os.path.join(run, "frames.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            for tr in r["tracks"]:
                rid = root(tr["id"])
                if rid not in ids:
                    continue
                s = set(tr["src"])
                cnt[rid]["both" if len(s) > 1 else ("thermal" if "thermal" in s else "rgb")] += 1
    evs = [e for e in events if cnt.get(e["id"])]
    fig, ax = plt.subplots(figsize=(11, 0.55 * max(2, len(evs)) + 1.5))
    for k, e in enumerate(evs):
        c = cnt[e["id"]]
        tot = sum(c.values())
        left = 0.0
        for key, col in (("thermal", S.CHANNEL_HEX["thermal"]), ("both", BOTH_HEX), ("rgb", S.CHANNEL_HEX["rgb"])):
            w = c[key] / tot
            if w > 0:
                ax.barh(k, w, left=left, color=col, height=0.56, edgecolor=S.PANEL, lw=2)
                if w > 0.08:
                    ax.text(left + w / 2, k, f"{w:.0%}", ha="center", va="center", fontsize=9.5,
                            color=S.BLACK if key != "both" else S.WHITE, fontweight="bold")
            left += w
    ax.set_yticks(range(len(evs)))
    ax.set_yticklabels([f"#{e['id']} {(e['class'] or e['category']).upper()}" for e in evs])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", visible=False)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in (S.CHANNEL_HEX["thermal"], BOTH_HEX, S.CHANNEL_HEX["rgb"])],
              ["THERMAL ONLY", "BOTH SENSORS", "VISIBLE ONLY"], loc="lower right", bbox_to_anchor=(1, 1.0), ncol=3, fontsize=9.5)
    _title(ax, "Sensor contribution", "share of each event's frames in which each channel detected the motion")
    _save(fig, path)


def zone_table(events: List[dict], ref_size, grid, fps: float) -> np.ndarray:
    cols, rows = grid
    rw, rh = ref_size
    z = np.zeros((rows, cols), np.float32)
    for e in events:
        pts = e["path"]
        for a, b in zip(pts[:-1], pts[1:]):
            dt = (b[2] - a[2]) / fps
            c = min(cols - 1, max(0, int(a[0] / rw * cols)))
            r = min(rows - 1, max(0, int(a[1] / rh * rows)))
            z[r, c] += dt
    return z


def chart_zones(z: np.ndarray, path: str) -> None:
    rows, cols = z.shape
    fig, ax = plt.subplots(figsize=(1.7 * cols + 1.4, 1.15 * rows + 1.3))
    cm = LinearSegmentedColormap.from_list("z", S.HEAT_RAMP[1:6])
    cm.set_bad(S.PANEL)
    ax.imshow(np.ma.masked_equal(z, 0), cmap=cm, vmin=0, vmax=max(1e-6, z.max()))
    for r in range(rows):
        for c in range(cols):
            v = z[r, c]
            ax.text(c, r, f"{v:.1f}s" if v > 0 else "·", ha="center", va="center", fontsize=13,
                    fontfamily="JetBrains Mono", color=S.WHITE if v > 0 else S.SILVER_3)
    for r in range(rows + 1):
        ax.axhline(r - 0.5, color=S.EDGE, lw=1)
    for c in range(cols + 1):
        ax.axvline(c - 0.5, color=S.EDGE, lw=1)
    ax.set_xticks(range(cols))
    ax.set_xticklabels([chr(65 + c) for c in range(cols)], fontfamily="JetBrains Mono")
    ax.set_yticks(range(rows))
    ax.set_yticklabels([str(r + 1) for r in range(rows)], fontfamily="JetBrains Mono")
    ax.tick_params(length=0)
    ax.grid(False)
    _title(ax, "Where · zone occupancy", "motion-seconds per zone (A1 = top-left)")
    _save(fig, path)


# ======================================================================================
# WHAT / SYSTEM
# ======================================================================================

def chart_categories(events: List[dict], path: str) -> None:
    cnt = Counter(e["category"] for e in events)
    dur = defaultdict(float)
    for e in events:
        dur[e["category"]] += e["duration_s"]
    cats = sorted(cnt, key=lambda c: -cnt[c]) or ["none"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 0.6 * max(2, len(cats)) + 1.6))
    for ax, vals, title, fmt in ((axs[0], [cnt[c] for c in cats], "What · events per category", "{:.0f}"),
                                 (axs[1], [dur[c] for c in cats], "Motion time per category", "{:.1f}s")):
        ax.barh(range(len(cats)), vals, color=[S.category_hex(c) for c in cats], height=0.56, edgecolor=S.PANEL, lw=2)
        for i, v in enumerate(vals):
            ax.text(v, i, "  " + fmt.format(v), va="center", color=S.WHITE, fontsize=11, fontfamily="JetBrains Mono")
        ax.set_yticks(range(len(cats)))
        ax.set_yticklabels([c.upper() for c in cats])
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        _title(ax, title)
        ax.set_xlim(0, max(vals + [1]) * 1.3)
    fig.tight_layout()
    _save(fig, path)


def chart_headings(events: List[dict], path: str) -> None:
    hs = [e["heading_deg"] for e in events if e.get("heading_deg") is not None]
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_facecolor(S.PANEL)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)  # image coordinates: +y is down
    bins = np.linspace(0, 2 * np.pi, 17)
    if hs:
        h, _ = np.histogram(np.deg2rad(hs), bins=bins)
        ax.bar(bins[:-1] + np.pi / 16, h, width=2 * np.pi / 16 * 0.86, color=S.ACCENT, edgecolor=S.PANEL, lw=1, alpha=0.9)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(["E · RIGHT", "S · DOWN", "W · LEFT", "N · UP"], fontfamily="JetBrains Mono", fontsize=8.5)
    ax.tick_params(colors=S.SILVER_2)
    ax.grid(color=S.GRID)
    ax.spines["polar"].set_color(S.EDGE)
    ax.set_title("HEADING ROSE", fontdict=TITLE_FONT, pad=18)
    _save(fig, path)


def chart_performance(meta: dict, path: str) -> None:
    st = meta["timing"]["stages"]
    names = [k for k in st if st[k]["calls"] > 0]
    names.sort(key=lambda k: -st[k]["mean_ms"])
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(names) + 1.5))
    vals = [st[k]["mean_ms"] for k in names]
    p95 = [st[k]["p95_ms"] for k in names]
    ax.barh(range(len(names)), p95, color=S.GRID, height=0.62)
    ax.barh(range(len(names)), vals, color=[S.ACCENT if i == 0 else S.SILVER_2 for i in range(len(names))], height=0.36)
    for i, k in enumerate(names):
        ax.text(max(vals[i], p95[i]), i, f"  {vals[i]:.1f} ms  ·  p95 {p95[i]:.0f}  ·  n={st[k]['calls']}",
                va="center", fontsize=9.5, fontfamily="JetBrains Mono", color=S.SILVER)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("_", " ").upper() for n in names])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(p95 + [1]) * 1.75)
    ax.set_xlabel("latency per call (ms) · bar = mean · shadow = p95")
    _title(ax, "System · stage latency", f"{meta['device']} · overall {meta['timing']['throughput_fps']} fps end-to-end")
    _save(fig, path)


# ======================================================================================
# HTML
# ======================================================================================

def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _img_tag(path: str, alt: str) -> str:
    return f'<img src="data:image/png;base64,{_b64(path)}" alt="{html.escape(alt)}">'


def _compass(deg) -> str:
    if deg is None:
        return "—"
    names = ["→ E", "↘ SE", "↓ S", "↙ SW", "← W", "↖ NW", "↑ N", "↗ NE"]
    return names[int(((deg + 22.5) % 360) // 45)]


CSS = """
@font-face{font-family:'Orbitron';src:url(data:font/ttf;base64,__ORB__) format('truetype');font-weight:400 900}
@font-face{font-family:'Rajdhani';src:url(data:font/ttf;base64,__RAJ__) format('truetype');font-weight:600}
:root{--bg:#0A0B0D;--panel:#111317;--card:#171A1F;--edge:#343841;--grid:#22252B;--ink:#CDD3DB;--ink2:#8E95A0;
--ink3:#5C626C;--white:#F1F3F6;--red:#C9A54E;--red2:#E8C76F;--deep:#3A3120}
*{box-sizing:border-box}
html{background:var(--bg)}
body{margin:0;color:var(--ink);font:500 16px/1.5 'Rajdhani',system-ui,sans-serif;letter-spacing:.2px;
background:radial-gradient(1200px 500px at 85% -10%,rgba(201,165,78,.10),transparent 60%),
linear-gradient(var(--bg),var(--bg));}
body:before{content:"";position:fixed;inset:0;pointer-events:none;opacity:.35;
background-image:linear-gradient(rgba(255,255,255,.025) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.025) 1px,transparent 1px);
background-size:32px 32px}
main{position:relative;max-width:1220px;margin:0 auto;padding:34px 16px 70px}
.brand{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.brand .bar{width:4px;height:36px;background:var(--red)}
h1{font-family:'Orbitron',sans-serif;font-weight:800;font-size:30px;letter-spacing:3px;margin:0;color:var(--white)}
.tag{font-family:'Orbitron';font-size:11px;letter-spacing:3px;color:var(--red2);border:1px solid var(--deep);padding:4px 10px}
.sub{color:var(--ink2);margin:10px 0 26px;font-size:15px}
code,.mono{font-family:'JetBrains Mono',ui-monospace,Menlo,Consolas,monospace;font-size:13px}
code{color:var(--white);background:var(--card);border:1px solid var(--edge);padding:1px 6px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.tile{position:relative;background:linear-gradient(180deg,var(--card),var(--panel));border:1px solid var(--edge);padding:14px 16px 12px}
.tile:before{content:"";position:absolute;left:-1px;top:-1px;width:28px;height:2px;background:var(--red)}
.tile b{display:block;font-family:'Orbitron';font-size:26px;font-weight:700;color:var(--white)}
.tile span{color:var(--ink2);font-size:13px;letter-spacing:1px;text-transform:uppercase}
.tile.hot b{color:var(--red2);text-shadow:0 0 14px rgba(232,199,111,.30)}
h2{display:flex;align-items:center;gap:12px;font-family:'Orbitron';font-size:15px;letter-spacing:3px;color:var(--white);margin:48px 0 14px;text-transform:uppercase}
h2 .n{color:var(--red2);font-size:12px}
h2:after{content:"";flex:1;height:1px;background:linear-gradient(90deg,var(--edge),transparent)}
.frame{position:relative;border:1px solid var(--edge);background:var(--panel);padding:6px;margin:10px 0}
.frame:before,.frame:after{content:"";position:absolute;width:14px;height:14px;border-color:var(--red);border-style:solid}
.frame:before{left:-1px;top:-1px;border-width:1px 0 0 1px}.frame:after{right:-1px;bottom:-1px;border-width:0 1px 1px 0}
img{max-width:100%;height:auto;display:block}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:14px}
.tw{overflow-x:auto;border:1px solid var(--edge);background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:14.5px}
th,td{padding:9px 12px;border-bottom:1px solid var(--grid);text-align:left;white-space:nowrap}
th{font-family:'Orbitron';font-size:10px;letter-spacing:2px;color:var(--ink2);font-weight:600;background:var(--card)}
tr:hover td{background:rgba(201,165,78,.06)}
.dot{display:inline-block;width:9px;height:9px;margin-right:8px;vertical-align:middle}
.chip{font-family:'JetBrains Mono';font-size:11px;padding:2px 7px;border:1px solid var(--edge);color:var(--ink)}
.chip.t{border-color:var(--deep);color:var(--red2)}
.note{color:var(--ink3);font-size:13px}
footer{margin-top:50px;color:var(--ink3);font-size:13px;border-top:1px solid var(--grid);padding-top:14px}
"""

HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>DualSight report</title>
<style>{css}</style></head><body><main>
<div class="brand"><div class="bar"></div><h1>DUALSIGHT</h1><span class="tag">MOTION INTELLIGENCE REPORT</span></div>
<p class="sub">{subtitle}</p>
<div class="tiles">{tiles}</div>
<h2><span class="n">01</span>When</h2><div class="frame">{timeline}</div>
<h2><span class="n">02</span>Where</h2>
<div class="frame">{traj}</div>
<div class="grid2"><div class="frame">{heat_rgb}</div><div class="frame">{heat_th}</div></div>
<div class="grid2"><div class="frame">{zones}</div><div class="frame">{headings}</div></div>
<h2><span class="n">03</span>What &amp; who saw it</h2><div class="frame">{cats}</div><div class="frame">{sensors}</div>
<h2><span class="n">04</span>Event log</h2><div class="tw"><table><thead><tr><th>#</th><th>Category</th><th>Class</th><th>Conf.</th>
<th>Start</th><th>End</th><th>Duration</th><th>Seen by</th><th>Heading</th><th>Speed px/s</th><th>Zone path</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<h2><span class="n">05</span>System</h2><div class="frame">{perf}</div><p class="sub">{pipeline}</p>
<footer>Generated by DualSight · published models only (YOLO11 · Real-ESRGAN · RAFT/DIS · OpenCV) · fonts: Orbitron, Rajdhani (SIL OFL)</footer>
</main></body></html>"""


def build_report(run: str, out_dir: str | None = None) -> str:
    frames, events, meta = _load(run)
    out = out_dir or os.path.join(run, "report")
    os.makedirs(out, exist_ok=True)
    cfg = meta["config"]
    ref_size = tuple(meta["streams"]["rgb_size"])
    fps = float(meta["streams"]["fps"])
    grid = cfg["analytics"]["grid"]

    p = {k: os.path.join(out, f"{k}.png") for k in
         ("timeline", "heatmap_rgb", "heatmap_thermal", "trajectories", "sensors", "zones", "categories", "headings",
          "performance")}
    chart_timeline(frames, events, p["timeline"], cfg["thermal"]["gate"])
    has_rgb = chart_heatmap(run, "rgb", "Event heat · visible", p["heatmap_rgb"], ref_size,
                            event_heat(run, events, ref_size, "rgb"))
    has_th = chart_heatmap(run, "thermal", "Event heat · thermal", p["heatmap_thermal"],
                           ref_size, event_heat(run, events, ref_size, "thermal"), meta.get("registration_thermal_to_rgb"))
    chart_trajectories(run, events, ref_size, p["trajectories"])
    chart_sensors(run, events, p["sensors"])
    z = zone_table(events, ref_size, grid, fps)
    chart_zones(z, p["zones"])
    chart_categories(events, p["categories"])
    chart_headings(events, p["headings"])
    chart_performance(meta, p["performance"])

    with open(os.path.join(out, "events.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "category", "class", "confidence", "start_s", "end_s", "duration_s", "seen_by",
                    "heading_deg", "mean_speed_px_s", "x1", "y1", "x2", "y2"])
        for e in events:
            w.writerow([e["id"], e["category"], e["class"], e["confidence"], e["start_s"], e["end_s"], e["duration_s"],
                        "+".join(e["seen_by"]), e["heading_deg"], e["mean_speed_px_s"], *e["max_box"]])

    n = len(frames)
    dur = float(frames["t"].iloc[-1] - frames["t"].iloc[0]) if n else 0.0
    by_cat = Counter(e["category"] for e in events)
    tiles = [(f"{len(events):02d}", "motion events", True),
             *[(f"{v:02d}", f"{k}", False) for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1])],
             (f"{dur:.0f}s", f"{n} synced frame pairs", False),
             (f"{100 * frames['fog_on'].mean():.0f}%", "defog active", False),
             (f"{100 * frames['deblur_on'].mean():.0f}%", "deblur active", False),
             (f"{meta['timing']['throughput_fps']}", f"fps · {meta['device']}", False)]
    tiles_html = "".join(f'<div class="tile{" hot" if hot else ""}"><b>{html.escape(a)}</b><span>{html.escape(b)}</span></div>'
                         for a, b, hot in tiles)

    cols, rows_n = grid
    rw, rh = ref_size
    rows = []
    for e in events:
        zs = []
        for q in e["path"]:
            zname = f"{chr(65 + min(cols - 1, int(q[0] / rw * cols)))}{1 + min(rows_n - 1, int(q[1] / rh * rows_n))}"
            if not zs or zs[-1] != zname:
                zs.append(zname)
        c = S.category_hex(e["category"])
        seen = " ".join(f"<span class='chip{' t' if s == 'thermal' else ''}'>{'THERMAL' if s == 'thermal' else 'VISIBLE'}</span>"
                        for s in e["seen_by"])
        rows.append(
            f"<tr><td class='mono'>{e['id']:03d}</td><td><span class='dot' style='background:{c};color:{c}'></span>"
            f"{html.escape(e['category']).upper()}</td><td>{html.escape(e['class'] or '—')}</td>"
            f"<td class='mono'>{e['confidence']:.2f}</td><td class='mono'>{_fmt_t(e['start_s'])}</td>"
            f"<td class='mono'>{_fmt_t(e['end_s'])}</td><td class='mono'>{e['duration_s']:.1f}s</td><td>{seen}</td>"
            f"<td class='mono'>{_compass(e['heading_deg'])}</td><td class='mono'>{e['mean_speed_px_s']:.0f}</td>"
            f"<td class='mono'>{' → '.join(zs[:8])}</td></tr>")
    b = meta["backends"]
    pipeline = (f"thermal enhancer <code>{b['thermal_enhancer']}</code> · optical flow <code>{b['optical_flow']}</code> · "
                f"detector <code>{b['detector']}</code> · ingest <code>{b['ingest']['thermal']}/{b['ingest']['rgb']}</code> · "
                f"sync violations {meta['streams']['sync_violations']} · dropped {meta['streams']['dropped']}")
    sub = (f"CH1 thermal <code>{html.escape(os.path.basename(meta['inputs']['thermal']))}</code> "
           f"{meta['streams']['thermal_size'][0]}×{meta['streams']['thermal_size'][1]} · CH2 visible "
           f"<code>{html.escape(os.path.basename(meta['inputs']['rgb']))}</code> "
           f"{meta['streams']['rgb_size'][0]}×{meta['streams']['rgb_size'][1]} · {fps:.1f} fps · v{meta['dualsight_version']}")
    css = CSS
    try:
        css = css.replace("__ORB__", _b64(S.FONTS["display"])).replace("__RAJ__", _b64(S.FONTS["ui"]))
    except OSError:
        css = css.replace("__ORB__", "").replace("__RAJ__", "")
    page = HTML.format(
        css=css, subtitle=sub, tiles=tiles_html, timeline=_img_tag(p["timeline"], "timeline"),
        heat_rgb=_img_tag(p["heatmap_rgb"], "visible heatmap") if has_rgb else "<p class='note'>no visible-channel events</p>",
        heat_th=_img_tag(p["heatmap_thermal"], "thermal heatmap") if has_th else "<p class='note'>no thermal-channel events</p>",
        traj=_img_tag(p["trajectories"], "trajectories"), zones=_img_tag(p["zones"], "zones"),
        headings=_img_tag(p["headings"], "headings"), cats=_img_tag(p["categories"], "categories"),
        sensors=_img_tag(p["sensors"], "sensor contribution"),
        rows="".join(rows) or "<tr><td colspan=11>no events</td></tr>",
        perf=_img_tag(p["performance"], "performance"), pipeline=pipeline)
    idx = os.path.join(out, "index.html")
    with open(idx, "w") as f:
        f.write(page)
    return idx
