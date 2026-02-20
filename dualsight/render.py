"""Offline visualization: rebuild the annotated dashboard from a run directory and the inputs.

The run directory stores *decisions* (gate states, camera shifts, tracks, final event labels),
not pixels, so a run can be re-rendered at any time - e.g. with the final, vote-settled labels
instead of the provisional live ones, or as a before/after comparison of the enhancement stages.
"""
from __future__ import annotations

import copy
import json
import logging
import os
from collections import Counter
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

from .enhance.deblur import MotionDeblur
from .enhance.thermal import ThermalEnhancer
from .io.sources import SyncedDualSource
from .io.writers import VideoWriter
from .utils.device import resolve_device
from .viz.overlay import Dashboard, FrameState, TrackView

log = logging.getLogger("dualsight")


def _label(img: np.ndarray, tag: str, text: str, active: bool = False) -> np.ndarray:
    """Theme plate: red index tag + silver caption; red frame corners when the stage is active."""
    from PIL import Image, ImageDraw

    from .viz import style as S

    out = img.copy()
    h, w = out.shape[:2]
    f_tag, f_txt = S.pil_font("display", 11), S.pil_font("ui_bold", 16)
    tw = int(f_tag.getlength(tag)) + int(f_txt.getlength(text)) + 34
    roi = out[0:30, 0:tw]
    roi[:] = (roi * 0.2 + np.array(S.hex_to_bgr(S.BLACK)) * 0.8).astype(np.uint8)
    cv2.rectangle(out, (0, 0), (3, 29), S.hex_to_bgr(S.ACCENT), -1)
    col = S.hex_to_bgr(S.ACCENT_BRIGHT if active else S.EDGE)
    L = 18
    for (cx, cy, dx, dy) in ((0, 0, 1, 1), (w - 1, 0, -1, 1), (0, h - 1, 1, -1), (w - 1, h - 1, -1, -1)):
        cv2.line(out, (cx, cy), (cx + dx * L, cy), col, 2)
        cv2.line(out, (cx, cy), (cx, cy + dy * L), col, 2)
    pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    d.text((10, 8), tag, font=f_tag, fill=S.hex_to_rgb(S.ACCENT))
    d.text((18 + f_tag.getlength(tag), 5), text, font=f_txt, fill=S.hex_to_rgb(S.WHITE if active else S.SILVER))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def render(run: str, thermal: Optional[str] = None, rgb: Optional[str] = None, out: Optional[str] = None,
           view: str = "enhanced", compare: bool = False, show: bool = False, with_report: bool = True,
           max_frames: int = 0) -> List[str]:
    with open(os.path.join(run, "run.json")) as f:
        meta = json.load(f)
    cfg = copy.deepcopy(meta["config"])
    thermal = thermal or meta["inputs"]["thermal"]
    rgb = rgb or meta["inputs"]["rgb"]
    frames = pd.read_csv(os.path.join(run, "frames.csv")).set_index("frame")
    with open(os.path.join(run, "events.json")) as f:
        events = {e["id"]: e for e in json.load(f)}
    merges: Dict[int, int] = {}
    mp = os.path.join(run, "track_merges.json")
    if os.path.exists(mp):
        with open(mp) as f:
            merges = {int(k): int(v) for k, v in json.load(f).items()}

    def root(tid: int) -> int:
        while tid in merges:
            tid = merges[tid]
        return tid

    per_frame: Dict[int, dict] = {}
    with open(os.path.join(run, "frames.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            per_frame[r["f"]] = r

    cfg["sources"]["realtime"] = False
    if max_frames:
        cfg["sources"]["max_frames"] = max_frames
    device = resolve_device(cfg.get("device", "auto"))
    src = SyncedDualSource(thermal, rgb, cfg)
    fps = src.fps
    tw, th = src.thermal.width, src.thermal.height
    rw, rh = src.rgb.width, src.rgb.height
    H = np.array(meta["registration_thermal_to_rgb"], np.float64)

    enh = deb = None
    if view == "enhanced" or compare:
        tcfg = copy.deepcopy(cfg["thermal"])
        tcfg["enhancer"] = meta["backends"]["thermal_enhancer"]
        enh = ThermalEnhancer(tcfg, device)
        deb = MotionDeblur(cfg["rgb"]["deblur"], device)

    dash = Dashboard(cfg["viz"], (rw, rh), (tw, th), H, fps)
    out = out or os.path.join(run, "render.mp4")
    writer = VideoWriter(out, fps)
    cmp_path = os.path.join(os.path.dirname(out), "compare.mp4")
    cmp_writer = VideoWriter(cmp_path, fps) if compare else None
    written = [out] + ([cmp_path] if compare else [])

    trails: Dict[int, list] = {}
    try:
        for pair in src:
            i = pair.index
            if i not in frames.index:
                break
            row = frames.loc[i]
            if enh is not None and int(row["fog_on"]) and (i == 0 or not int(frames.loc[i - 1]["fog_on"])):
                enh.reset()
            th_raw = pair.thermal if pair.thermal.ndim == 2 else cv2.cvtColor(pair.thermal, cv2.COLOR_BGR2GRAY)
            th_img, rgb_img = th_raw, pair.rgb
            if enh is not None and int(row["fog_on"]):
                shift = None
                if int(row.get("cam_reliable", 0)):
                    shift = (float(row["cam_dx"]) / H[0, 0], float(row["cam_dy"]) / H[1, 1])
                th_enh = enh(th_raw, float(row["fog_score"]), shift)
                th_enh = cv2.resize(th_enh, (tw, th)) if th_enh.shape != th_raw.shape else th_enh
            else:
                th_enh = th_raw
            if deb is not None and int(row["deblur_on"]):
                rgb_deb = deb(pair.rgb, float(row["cam_dx"]), float(row["cam_dy"]))
            else:
                rgb_deb = pair.rgb
            if view == "enhanced":
                th_img, rgb_img = th_enh, rgb_deb

            rec = per_frame.get(i, {"tracks": []})
            tvs = []
            seen = set()
            for tr in rec["tracks"]:
                tr = dict(tr, id=root(tr["id"]))
                if tr["id"] in seen:
                    continue  # two fragments of one merged event in the same frame -> draw once
                seen.add(tr["id"])
                ev = events.get(tr["id"])
                if ev is None:
                    continue  # not a final event (noise / too short) -> hidden in the clean render
                c = ((tr["box"][0] + tr["box"][2]) / 2, (tr["box"][1] + tr["box"][3]) / 2)
                trails.setdefault(tr["id"], []).append(c)
                trails[tr["id"]] = trails[tr["id"]][-45:]
                tvs.append(TrackView(tr["id"], tr["box"], ev["category"], ev["class"], ev["confidence"],
                                     tr["src"], ev["heading_deg"], list(trails[tr["id"]])))
            started = Counter(e["category"] for e in events.values() if e["start_frame"] <= i)
            st = FrameState(i, pair.t, bool(row["fog_on"]), float(row["fog_score"]), bool(row["deblur_on"]),
                            float(row["cam_shift"]), float(row["motion_thermal"]), float(row["motion_rgb"]), tvs,
                            None, None, meta["backends"]["thermal_enhancer"] if view == "enhanced" else "off",
                            dict(started), float(row["cam_dx"]), float(row["cam_dy"]), 0.0,
                            bool(row.get("transition", 0)))
            canvas = dash.compose(th_img, rgb_img, st)
            writer.write(canvas)
            if cmp_writer is not None:
                pw = 640
                ph = int(round(pw * rh / rw))
                tr_ = lambda g: cv2.cvtColor(cv2.resize(g, (pw, ph)), cv2.COLOR_GRAY2BGR)
                fog, blur = bool(int(row["fog_on"])), bool(int(row["deblur_on"]))
                gap = np.full((ph, 6, 3), 7, np.uint8)
                top = np.hstack([_label(tr_(th_raw), "CH1 IN", "THERMAL  RAW"), gap,
                                 _label(tr_(th_enh), "CH1 OUT", "DEFOGGED" if fog else "GATE CLEAR · PASSTHROUGH", fog)])
                bot = np.hstack([_label(cv2.resize(pair.rgb, (pw, ph)), "CH2 IN", "VISIBLE  RAW"), gap,
                                 _label(cv2.resize(rgb_deb, (pw, ph)), "CH2 OUT", "DEBLURRED" if blur else "CAMERA STILL · PASSTHROUGH", blur)])
                cmp_writer.write(np.vstack([top, np.full((6, top.shape[1], 3), 7, np.uint8), bot]))
            if show:
                cv2.imshow("DualSight render", canvas)
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    break
    finally:
        src.release()
        writer.release()
        if cmp_writer is not None:
            cmp_writer.release()
        if show:
            cv2.destroyAllWindows()
    if with_report:
        from .analytics.report import build_report

        written.append(build_report(run))
    return written
