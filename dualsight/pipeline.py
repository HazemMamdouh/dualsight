"""End-to-end orchestration of the dual-channel pipeline.

    thermal ─► FogGate ──(fog)──► ThermalEnhancer ─┐
                       └(clear)──────────────────► ├─► MotionDetector(thermal) ─┐
    rgb ─────► CameraMotionGate ─(moving)─► Deblur ┤                            ├─► fuse ─► track ─► attribute (YOLO on ROIs)
                                └(still)─────────► └─► MotionDetector(rgb) ─────┘                      │
                                                                                 analytics / heatmaps / live dashboard
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import platform
import time
from collections import Counter
from typing import Dict, List, Optional

import cv2
import numpy as np

from . import __version__
from .detect.detector import ObjectDetector, associate
from .enhance.deblur import MotionDeblur
from .enhance.quality import CameraMotionGate, FogGate
from .enhance.thermal import ThermalEnhancer
from .io.sources import SyncedDualSource
from .io.writers import VideoWriter
from .motion.detector import MotionDetector
from .motion.flow import FlowEstimator
from .motion.tracker import MotionTracker, Track, fuse_blobs, project_box
from .utils.device import resolve_device
from .utils.timing import StageTimer
from .viz.overlay import Dashboard, FrameState, TrackView

log = logging.getLogger("dualsight")

FRAME_FIELDS = ["frame", "t", "skew_ms", "dropped", "fog_score", "fog_spread", "fog_grad", "fog_on",
                "cam_dx", "cam_dy", "cam_shift", "cam_reliable", "deblur_on", "transition", "motion_thermal", "motion_rgb",
                "blobs_thermal", "blobs_rgb", "tracks_active", "detections", "proc_ms"]


def scale_h(sx: float, sy: float) -> np.ndarray:
    return np.diag([sx, sy, 1.0]).astype(np.float64)


def gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def track_summary(t: Track, fps: float) -> dict:
    cat, fine, conf = t.label()
    ks = sorted(t.boxes)
    path = [[round((t.boxes[k][0] + t.boxes[k][2]) / 2, 1), round((t.boxes[k][1] + t.boxes[k][3]) / 2, 1), k] for k in ks]
    step = max(1, len(path) // 60)
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in t.boxes.values()]
    dist = sum(math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]) for i in range(len(path) - 1))
    dur = (t.last_frame - t.first_frame + 1) / fps
    big = t.boxes[ks[int(np.argmax(areas))]]
    return {
        "id": t.id, "category": cat, "class": fine, "confidence": round(conf, 3),
        "start_frame": t.first_frame, "end_frame": t.last_frame,
        "start_s": round(t.first_frame / fps, 3), "end_s": round(t.last_frame / fps, 3), "duration_s": round(dur, 3),
        "sources": dict(t.sources), "seen_by": sorted(t.sources),
        "heading_deg": None if t.direction_deg() is None else round(t.direction_deg(), 1),
        "mean_speed_px_s": round(dist / max(1e-6, dur), 1),
        "votes": {k: round(v, 3) for k, v in t.votes.items()},
        "detections": t.det_hits, "max_box": [round(v, 1) for v in big],
        "path": path[::step] + ([path[-1]] if path and (len(path) - 1) % step else []),
    }


class Pipeline:
    def __init__(self, cfg: dict, out_dir: str, show: bool = False, save_video: bool = False,
                 save_processed: bool = False) -> None:
        self.cfg = cfg
        self.out = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.device = resolve_device(cfg["device"])
        self.show = show
        self.save_video = save_video
        self.save_processed = save_processed
        log.info("device: %s", self.device)

        self.fog_gate = FogGate(cfg["thermal"]["gate"])
        self.cam_gate = CameraMotionGate(cfg["rgb"]["gate"])
        self.enhancer = ThermalEnhancer(cfg["thermal"], self.device)
        self.deblur = MotionDeblur(cfg["rgb"]["deblur"], self.device)
        self.flow = FlowEstimator(cfg["motion"]["flow"], self.device)
        self.md = {ch: MotionDetector(ch, cfg["motion"], self.flow) for ch in ("thermal", "rgb")}
        self.detector: Optional[ObjectDetector] = None
        if cfg["detector"]["enabled"]:
            self.detector = ObjectDetector(cfg["detector"], self.device)
        self.timer = StageTimer()

    # ------------------------------------------------------------------------------------
    def run(self, thermal_spec: str, rgb_spec: str) -> dict:
        cfg = self.cfg
        src = SyncedDualSource(thermal_spec, rgb_spec, cfg)
        fps = src.fps
        tw, th = src.thermal.width, src.thermal.height
        rw, rh = src.rgb.width, src.rgb.height
        ref_size = (rw, rh)
        H_reg = np.array(cfg["registration"]["thermal_to_rgb"], np.float64) if cfg["registration"]["thermal_to_rgb"] \
            else scale_h(rw / tw, rh / th)
        pw = int(cfg["motion"]["proc_width"])
        proc = {"thermal": (pw, int(round(pw * th / tw))), "rgb": (pw, int(round(pw * rh / rw)))}
        to_ref = {"thermal": H_reg @ scale_h(tw / proc["thermal"][0], th / proc["thermal"][1]),
                  "rgb": scale_h(rw / proc["rgb"][0], rh / proc["rgb"][1])}
        ch_to_ref = {"thermal": H_reg, "rgb": np.eye(3)}
        hm_w = int(cfg["analytics"]["heatmap_width"])
        hm_h = int(round(hm_w * rh / rw))
        hm_from_proc = {ch: scale_h(hm_w / rw, hm_h / rh) @ to_ref[ch] for ch in to_ref}
        heat = {ch: np.zeros((hm_h, hm_w), np.float32) for ch in ("thermal", "rgb", "fused")}

        tracker = MotionTracker(cfg["tracking"], ref_size)
        dash = Dashboard(cfg["viz"], ref_size, (tw, th), H_reg, fps) if (self.show or self.save_video) else None
        writer = VideoWriter(os.path.join(self.out, "live.mp4"), fps) if self.save_video else None
        w_proc = None
        if self.save_processed:
            w_proc = {"thermal": VideoWriter(os.path.join(self.out, "thermal_enhanced.mp4"), fps),
                      "rgb": VideoWriter(os.path.join(self.out, "rgb_deblurred.mp4"), fps)}

        f_csv = open(os.path.join(self.out, "frames.csv"), "w", newline="")
        csv_w = csv.DictWriter(f_csv, fieldnames=FRAME_FIELDS)
        csv_w.writeheader()
        f_jsonl = open(os.path.join(self.out, "frames.jsonl"), "w")
        f_det = open(os.path.join(self.out, "detections.csv"), "w", newline="")
        det_w = csv.writer(f_det)
        det_w.writerow(["frame", "t", "channel", "class", "category", "conf", "x1", "y1", "x2", "y2", "track_id"])

        bg_samples: Dict[str, List[np.ndarray]] = {"thermal": [], "rgb": []}
        sample_every = max(1, int((src.expected_frames or 600) / 40))
        prev_fog_state = False
        from collections import deque
        fog_hist: deque = deque(maxlen=max(2, int(round(fps))))  # ~1 s window
        trails: Dict[int, List] = {}
        live_events: Counter = Counter()
        counted: set = set()
        wall0 = time.perf_counter()
        n = 0
        win = "DualSight" if self.show else None
        log.info("processing at %dx%d (thermal) / %dx%d (rgb), ref %dx%d",
                 *proc["thermal"], *proc["rgb"], rw, rh)

        try:
            for pair in src:
                self.timer.start_frame()
                t0 = time.perf_counter()
                i = pair.index
                # ---------------- thermal: gate + enhancement
                # ---------------- rig ego-motion (measured on RGB corners, shared by both sensors)
                with self.timer.stage("rgb_gate"):
                    cam_on, cm = self.cam_gate(pair.rgb)
                hint = {}
                if cm.reliable and self.cfg["motion"]["ego"].get("share_rig_motion", True):
                    for ch in ("thermal", "rgb"):
                        sx, sy = 1.0 / to_ref[ch][0, 0], 1.0 / to_ref[ch][1, 1]
                        hint[ch] = np.float32([[1, 0, cm.dx * sx], [0, 1, cm.dy * sy]])
                th_shift = (cm.dx / H_reg[0, 0], cm.dy / H_reg[1, 1]) if cm.reliable else None
                # ---------------- thermal: gate + enhancement
                with self.timer.stage("thermal_gate"):
                    th_gray = gray(pair.thermal)
                    fog_on, fm = self.fog_gate(th_gray)
                if fog_on and not prev_fog_state:
                    self.enhancer.reset()
                with self.timer.stage("thermal_enhance"):
                    if fog_on:
                        th_img = self.enhancer(th_gray, fm["fog_score_smooth"], th_shift)
                        if th_img.shape != th_gray.shape:
                            th_img = cv2.resize(th_img, (tw, th), interpolation=cv2.INTER_AREA)
                    else:
                        th_img = th_gray
                if fog_on != prev_fog_state:
                    # appearance changes abruptly when enhancement toggles -> restart that channel's models
                    self.md["thermal"].reset()
                    prev_fog_state = fog_on
                # ---------------- atmospheric-transition guard: while fog rolls in/out the whole
                # scene radiance drifts and brightness constancy breaks, so blobs must be stronger
                fog_hist.append(fm["fog_score_smooth"])
                rate = abs(fog_hist[-1] - fog_hist[0]) / max(1e-6, (len(fog_hist) - 1) / fps) if len(fog_hist) > 1 else 0.0
                guard = cfg["motion"].get("transition_guard", {})
                in_transition = rate > float(guard.get("fog_rate", 1e9))
                boost = float(guard.get("snr_boost", 1.0)) if in_transition else 1.0
                # ---------------- rgb: deblur only while the camera moves
                with self.timer.stage("rgb_deblur"):
                    rgb_img = self.deblur(pair.rgb, cm.dx, cm.dy) if cam_on else pair.rgb
                # ---------------- motion per channel
                res = {}
                with self.timer.stage("motion_thermal"):
                    res["thermal"] = self.md["thermal"](cv2.resize(th_img, proc["thermal"], interpolation=cv2.INTER_AREA),
                                                        hint.get("thermal"), boost)
                with self.timer.stage("motion_rgb"):
                    res["rgb"] = self.md["rgb"](cv2.resize(gray(rgb_img), proc["rgb"], interpolation=cv2.INTER_AREA),
                                                hint.get("rgb"), boost)
                # ---------------- fusion + tracking
                with self.timer.stage("track"):
                    per = {}
                    for ch in ("thermal", "rgb"):
                        if ch not in cfg["motion"]["channels"]:
                            continue
                        S = to_ref[ch]
                        sx = S[0, 0]
                        per[ch] = [(project_box(b.box, S), [b.flow[0] * sx, b.flow[1] * sx], b.area * sx * sx)
                                   for b in res[ch].blobs]
                    fused = fuse_blobs(per)
                    active = tracker.update(i, fused)
                    for ch in ("thermal", "rgb"):
                        m = res[ch].mask
                        if m.any():
                            hm = cv2.warpPerspective(m, hm_from_proc[ch], (hm_w, hm_h), flags=cv2.INTER_NEAREST)
                            heat[ch] += (hm > 0)
                            heat["fused"] += (hm > 0) * 0.5
                # ---------------- attribution
                dets_out = []
                if self.detector is not None and active and i % int(cfg["detector"]["every"]) == 0:
                    with self.timer.stage("detect"):
                        regions = [t.box for t in active]
                        all_dets = []
                        for ch in cfg["detector"]["on_channels"]:
                            frame_ch = rgb_img if ch == "rgb" else th_img
                            Hc = ch_to_ref[ch]
                            all_dets += self.detector.detect(frame_ch, ch, regions, Hc, np.linalg.inv(Hc))
                        dets_out = associate(all_dets, active)
                        for d, tid in dets_out:
                            det_w.writerow([i, round(pair.t, 3), d.channel, d.cls, d.category, round(d.conf, 3),
                                            *[round(v, 1) for v in d.box], tid if tid is not None else ""])
                # ---------------- bookkeeping
                if i % sample_every == 0:
                    bg_samples["thermal"].append(th_img.copy())
                    bg_samples["rgb"].append(cv2.resize(rgb_img, (min(rw, 960), int(min(rw, 960) * rh / rw))))
                vis_tracks = []
                for t in active:
                    trails.setdefault(t.id, []).append(((t.box[0] + t.box[2]) / 2, (t.box[1] + t.box[3]) / 2))
                    trails[t.id] = trails[t.id][-45:]
                    if not t.confirmed:
                        continue
                    cat, fine, conf = t.label()
                    if t.id not in counted and tracker.is_event(t, fps):
                        counted.add(t.id)
                    vis_tracks.append(TrackView(t.id, list(t.boxes.get(i, t.box)), cat, fine, conf, sorted(t.sources),
                                                t.direction_deg(), list(trails[t.id])))
                live_events = Counter()
                for tt in list(tracker.tracks.values()) + tracker.finished:
                    if tt.id in counted:
                        live_events[tt.label()[0]] += 1
                proc_ms = (time.perf_counter() - t0) * 1000.0
                csv_w.writerow({
                    "frame": i, "t": round(pair.t, 4), "skew_ms": round(pair.skew_ms, 2), "dropped": pair.dropped,
                    "fog_score": round(fm["fog_score_smooth"], 4), "fog_spread": round(fm["spread"], 4),
                    "fog_grad": round(fm["grad"], 3), "fog_on": int(fog_on),
                    "cam_dx": round(cm.dx, 3), "cam_dy": round(cm.dy, 3), "cam_shift": round(cm.magnitude, 3),
                    "cam_reliable": int(cm.reliable),
                    "deblur_on": int(cam_on), "transition": int(in_transition), "motion_thermal": round(res["thermal"].score, 6),
                    "motion_rgb": round(res["rgb"].score, 6), "blobs_thermal": len(res["thermal"].blobs),
                    "blobs_rgb": len(res["rgb"].blobs), "tracks_active": len(vis_tracks),
                    "detections": len(dets_out), "proc_ms": round(proc_ms, 2)})
                f_jsonl.write(json.dumps({
                    "f": i, "t": round(pair.t, 4),
                    "tracks": [{"id": v.id, "box": [round(x, 1) for x in v.box], "src": list(v.sources),
                                "label": v.label, "cls": v.fine, "conf": round(v.conf, 3),
                                "heading": None if v.heading is None else round(v.heading, 1)} for v in vis_tracks],
                    "blobs": {ch: [[round(x, 1) for x in b[0]] for b in per.get(ch, [])] for ch in per},
                    "dets": [{"box": [round(x, 1) for x in d.box], "cls": d.cls, "conf": round(d.conf, 3),
                              "ch": d.channel, "track": tid} for d, tid in dets_out],
                }) + "\n")

                if w_proc is not None:
                    w_proc["thermal"].write(th_img)
                    w_proc["rgb"].write(rgb_img)
                if dash is not None:
                    with self.timer.stage("render"):
                        st = FrameState(i, pair.t, fog_on, fm["fog_score_smooth"], cam_on, cm.magnitude,
                                        res["thermal"].score, res["rgb"].score, vis_tracks,
                                        res["thermal"].mask, res["rgb"].mask, self.enhancer.method, dict(live_events),
                                        cm.dx, cm.dy, n / max(1e-6, time.perf_counter() - wall0), in_transition)
                        canvas = dash.compose(th_img, rgb_img, st)
                    if writer is not None:
                        writer.write(canvas)
                    if win is not None:
                        cv2.imshow(win, canvas)
                        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                            log.info("stopped by user")
                            break
                n += 1
                if n % 50 == 0:
                    el = time.perf_counter() - wall0
                    tot = src.expected_frames
                    log.info("frame %d%s | %.1f fps | fog=%s deblur=%s | tracks=%d | events=%s", n,
                             f"/{tot}" if tot else "", n / el, int(fog_on), int(cam_on), len(vis_tracks), dict(live_events))
        finally:
            src.release()
            for f in (f_csv, f_jsonl, f_det):
                f.close()
            if writer is not None:
                writer.release()
            if w_proc is not None:
                for w in w_proc.values():
                    w.release()
            if win is not None:
                cv2.destroyAllWindows()

        elapsed = time.perf_counter() - wall0
        tracks = tracker.consolidate(tracker.close(), fps)
        events = [track_summary(t, fps) for t in tracks if tracker.is_event(t, fps)]
        with open(os.path.join(self.out, "track_merges.json"), "w") as f:
            json.dump({str(k): v for k, v in tracker.merged_into.items()}, f)
        events.sort(key=lambda e: e["start_s"])
        with open(os.path.join(self.out, "events.json"), "w") as f:
            json.dump(events, f, indent=1)
        for ch, hm in heat.items():
            np.save(os.path.join(self.out, f"heatmap_{ch}.npy"), hm)
        for ch, samples in bg_samples.items():
            if samples:
                cv2.imwrite(os.path.join(self.out, f"background_{ch}.png"), np.median(np.stack(samples), axis=0).astype(np.uint8))

        meta = {
            "dualsight_version": __version__,
            "inputs": {"thermal": str(thermal_spec), "rgb": str(rgb_spec)},
            "device": self.device, "host": platform.platform(), "python": platform.python_version(),
            "backends": {"thermal_enhancer": self.enhancer.method, "optical_flow": self.flow.backend,
                         "detector": cfg["detector"]["model"] if self.detector else None,
                         "ingest": {"thermal": src.thermal.backend, "rgb": src.rgb.backend}},
            "streams": {"fps": fps, "thermal_size": [tw, th], "rgb_size": [rw, rh],
                        "frames": n, "sync_violations": src.n_skew_violations, "dropped": src.n_dropped},
            "registration_thermal_to_rgb": H_reg.tolist(),
            "timing": {"elapsed_s": round(elapsed, 2), "throughput_fps": round(n / max(1e-6, elapsed), 2),
                       "stages": self.timer.summary()},
            "events": {"total": len(events), "by_category": dict(Counter(e["category"] for e in events))},
            "config": cfg,
        }
        with open(os.path.join(self.out, "run.json"), "w") as f:
            json.dump(meta, f, indent=1, default=str)
        log.info("done: %d frames in %.1fs (%.2f fps) -> %d events %s", n, elapsed, n / max(1e-6, elapsed),
                 len(events), meta["events"]["by_category"])
        return meta
