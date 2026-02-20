"""Motion-triggered object attribution ("who/what made the motion?").

The detector never runs on idle frames. While motion tracks are active it runs
every ``every`` frames on the motion regions only (``mode: roi``): each region
is padded, squared and cropped, so a 40-px target in a 1280-px frame is seen at
a useful scale. Detections are mapped back to reference coordinates and vote
into the overlapping motion track; the track label is the vote winner.

Any Ultralytics detection checkpoint works. With the published COCO models the
80 classes are folded into the categories that matter for perimeter
monitoring (person / vehicle / animal / ...).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

log = logging.getLogger("dualsight")

CATEGORY_MAP: Dict[str, str] = {
    "person": "person",
    "bicycle": "vehicle", "car": "vehicle", "motorcycle": "vehicle", "bus": "vehicle",
    "truck": "vehicle", "train": "vehicle",
    "boat": "vessel", "airplane": "aircraft",
    "bird": "animal", "cat": "animal", "dog": "animal", "horse": "animal", "sheep": "animal",
    "cow": "animal", "elephant": "animal", "bear": "animal", "zebra": "animal", "giraffe": "animal",
    "backpack": "carried-object", "handbag": "carried-object", "suitcase": "carried-object",
}


@dataclass
class Detection:
    box: List[float]      # reference (RGB) pixel coordinates
    cls: str              # detector class name
    category: str         # folded category
    conf: float
    channel: str


def _square_roi(box: Sequence[float], pad: float, min_size: int, W: int, H: int) -> Tuple[int, int, int, int]:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    s = max(box[2] - box[0], box[3] - box[1]) * pad
    s = max(s, float(min_size))
    s = min(s, float(max(W, H)))
    x1 = int(round(min(max(0.0, cx - s / 2), max(0.0, W - s))))
    y1 = int(round(min(max(0.0, cy - s / 2), max(0.0, H - s))))
    return x1, y1, int(min(W, x1 + s)), int(min(H, y1 + s))


def _merge_rois(rois: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    rois = list(rois)
    changed = True
    while changed:
        changed = False
        out: List[Tuple[int, int, int, int]] = []
        for r in rois:
            for k, o in enumerate(out):
                ix = min(r[2], o[2]) - max(r[0], o[0])
                iy = min(r[3], o[3]) - max(r[1], o[1])
                if ix > 0 and iy > 0 and ix * iy > 0.3 * min((r[2] - r[0]) * (r[3] - r[1]), (o[2] - o[0]) * (o[3] - o[1])):
                    out[k] = (min(r[0], o[0]), min(r[1], o[1]), max(r[2], o[2]), max(r[3], o[3]))
                    changed = True
                    break
            else:
                out.append(r)
        rois = out
    return rois


class ObjectDetector:
    def __init__(self, cfg: dict, device: str) -> None:
        from ultralytics import YOLO

        self.cfg = cfg
        self.device = device
        self.model = YOLO(cfg["model"])
        self.names = self.model.names
        self.half = bool(cfg.get("half")) and device.startswith("cuda")
        log.info("object detector: %s on %s (mode=%s, every=%d)", cfg["model"], device, cfg["mode"], cfg["every"])

    def _predict(self, images: List[np.ndarray]):
        kw = {"half": True} if self.half else {}
        return self.model.predict(images, conf=float(self.cfg["conf"]), imgsz=int(self.cfg["imgsz"]),
                                  device=self.device, verbose=False, **kw)

    def detect(self, frame: np.ndarray, channel: str, regions: List[List[float]], to_ref: np.ndarray,
               from_ref: np.ndarray) -> List[Detection]:
        """Run on ``frame`` (channel pixels). ``regions`` are motion boxes in reference coordinates;
        ``to_ref`` / ``from_ref`` are 3x3 homographies channel->reference and back."""
        from ..motion.tracker import project_box

        H, W = frame.shape[:2]
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        crops, offsets = [], []
        if self.cfg["mode"] == "full" or len(regions) > int(self.cfg["max_rois"]):
            crops, offsets = [frame], [(0, 0)]
        else:
            ch_boxes = [project_box(r, from_ref) for r in regions]
            scale = 1.0 / max(1e-6, abs(float(to_ref[0, 0])))  # min ROI size is given in reference px
            rois = [_square_roi(b, float(self.cfg["roi_pad"]), int(self.cfg["roi_min"] * scale), W, H) for b in ch_boxes]
            for x1, y1, x2, y2 in _merge_rois(rois):
                if x2 - x1 < 8 or y2 - y1 < 8:
                    continue
                crops.append(frame[y1:y2, x1:x2])
                offsets.append((x1, y1))
        if not crops:
            return []
        dets: List[Detection] = []
        for res, (ox, oy) in zip(self._predict(crops), offsets):
            if res.boxes is None or len(res.boxes) == 0:
                continue
            for b, c, s in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                name = self.names[int(c)]
                cat = CATEGORY_MAP.get(name)
                if cat is None:
                    continue
                box = [float(b[0] + ox), float(b[1] + oy), float(b[2] + ox), float(b[3] + oy)]
                dets.append(Detection(project_box(box, to_ref), name, cat, float(s), channel))
        return dets


def associate(dets: List[Detection], tracks, min_overlap: float = 0.3) -> List[Tuple[Detection, Optional[int]]]:
    """Attach each detection to the motion track it overlaps most; returns (det, track_id|None)."""
    from ..motion.tracker import overlap_small

    out = []
    for d in dets:
        best, best_ov = None, min_overlap
        for t in tracks:
            ov = overlap_small(d.box, t.box)
            if ov > best_ov:
                best, best_ov = t, ov
        if best is not None:
            best.votes[d.category] += d.conf
            best.fine_votes[d.cls] += d.conf
            best.det_hits += 1
        out.append((d, None if best is None else best.id))
    return out
