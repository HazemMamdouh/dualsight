"""Cross-channel blob fusion and motion-event tracking.

Blobs from both channels are projected into one reference frame (RGB pixel
coordinates), merged when they overlap, and associated over time with a
Hungarian assignment on ``1 - IoU`` gated by centre distance. Each track keeps
which channel(s) saw it, its trajectory, its motion direction and the votes of
the object detector, and becomes a reportable *motion event* once it has been
confirmed for ``min_hits`` frames.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

Box = List[float]


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(1e-9, ua)


def overlap_small(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over the smaller box."""
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    sa = (a[2] - a[0]) * (a[3] - a[1])
    sb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(1e-9, min(sa, sb))


def _grow(b: Sequence[float], frac: float) -> List[float]:
    w, h = b[2] - b[0], b[3] - b[1]
    return [b[0] - frac * w, b[1] - frac * h, b[2] + frac * w, b[3] + frac * h]


@dataclass
class FusedBlob:
    box: Box
    sources: List[str]
    flow: List[float]       # reference px / frame
    area: float


def project_box(box: Box, H: np.ndarray) -> Box:
    pts = np.float32([[box[0], box[1], 1], [box[2], box[1], 1], [box[0], box[3], 1], [box[2], box[3], 1]]).T
    q = H @ pts
    q = q[:2] / q[2:3]
    return [float(q[0].min()), float(q[1].min()), float(q[0].max()), float(q[1].max())]


def fuse_blobs(per_channel: Dict[str, List[Tuple[Box, List[float], float]]], merge_overlap: float = 0.3) -> List[FusedBlob]:
    """Greedy union of overlapping boxes coming from different channels."""
    items: List[FusedBlob] = []
    for ch, blobs in per_channel.items():
        for box, flow, area in blobs:
            items.append(FusedBlob(list(box), [ch], list(flow), area))
    items.sort(key=lambda b: -b.area)
    merged: List[FusedBlob] = []
    for it in items:
        for m in merged:
            if overlap_small(m.box, it.box) >= merge_overlap:
                w_m, w_i = m.area, it.area
                m.flow = [(m.flow[k] * w_m + it.flow[k] * w_i) / (w_m + w_i) for k in range(2)]
                m.box = [min(m.box[0], it.box[0]), min(m.box[1], it.box[1]),
                         max(m.box[2], it.box[2]), max(m.box[3], it.box[3])]
                m.sources = sorted(set(m.sources) | set(it.sources))
                m.area += it.area
                break
        else:
            merged.append(it)
    return merged


@dataclass
class Track:
    id: int
    first_frame: int
    last_frame: int
    box: Box
    hits: int = 1
    missed: int = 0
    confirmed: bool = False
    boxes: Dict[int, Box] = field(default_factory=dict)
    sources: Counter = field(default_factory=Counter)
    flows: List[List[float]] = field(default_factory=list)
    votes: Counter = field(default_factory=Counter)        # coarse category -> summed confidence
    fine_votes: Counter = field(default_factory=Counter)   # detector class name -> summed confidence
    det_hits: int = 0
    vel: Tuple[float, float] = (0.0, 0.0)                  # px / frame, EMA of the box-centre motion

    def predicted(self, frame: int) -> Box:
        """Constant-velocity prediction - bridges short gaps (occlusion, fog patches, gate toggles)."""
        dt = frame - self.last_frame
        dx, dy = self.vel[0] * dt, self.vel[1] * dt
        return [self.box[0] + dx, self.box[1] + dy, self.box[2] + dx, self.box[3] + dy]

    @property
    def center(self) -> Tuple[float, float]:
        return (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2

    def label(self, min_score: float = 0.35) -> Tuple[str, str, float]:
        if not self.votes:
            return "unknown", "", 0.0
        cat, s = self.votes.most_common(1)[0]
        total = sum(self.votes.values())
        if s < min_score:
            return "unknown", "", 0.0
        fine = ""
        cand = [(n, v) for n, v in self.fine_votes.most_common() if v > 0]
        if cand:
            fine = cand[0][0]
        conf = min(1.0, s / max(1, self.det_hits)) * (s / total)
        return cat, fine, float(conf)

    def displacement(self) -> float:
        if len(self.boxes) < 2:
            return 0.0
        ks = sorted(self.boxes)
        a, b = self.boxes[ks[0]], self.boxes[ks[-1]]
        return math.hypot((b[0] + b[2] - a[0] - a[2]) / 2, (b[1] + b[3] - a[1] - a[3]) / 2)

    def direction_deg(self) -> Optional[float]:
        """Image-plane heading in degrees (0 = right, 90 = down), from the trajectory."""
        if len(self.boxes) < 3:
            return None
        ks = sorted(self.boxes)
        a, b = self.boxes[ks[0]], self.boxes[ks[-1]]
        dx = (b[0] + b[2] - a[0] - a[2]) / 2
        dy = (b[1] + b[3] - a[1] - a[3]) / 2
        if math.hypot(dx, dy) < 5:
            if not self.flows:
                return None
            f = np.mean(self.flows, axis=0)
            dx, dy = float(f[0]), float(f[1])
        return float(math.degrees(math.atan2(dy, dx)) % 360)


class MotionTracker:
    def __init__(self, cfg: dict, ref_size: Tuple[int, int]) -> None:
        self.cfg = cfg
        self.diag = math.hypot(*ref_size)
        self.tracks: Dict[int, Track] = {}
        self.finished: List[Track] = []
        self.merged_into: Dict[int, int] = {}
        self._next = 1

    def update(self, frame: int, blobs: List[FusedBlob]) -> List[Track]:
        active = [t for t in self.tracks.values()]
        C = np.ones((len(active), len(blobs)), np.float32)
        maxd = float(self.cfg["max_center_dist"]) * self.diag
        preds = [t.predicted(frame) for t in active]
        for i, t in enumerate(active):
            pb = preds[i]
            tc = ((pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2)
            for j, b in enumerate(blobs):
                bc = ((b.box[0] + b.box[2]) / 2, (b.box[1] + b.box[3]) / 2)
                d = math.hypot(tc[0] - bc[0], tc[1] - bc[1])
                ov = max(iou(pb, b.box), 0.5 * overlap_small(pb, b.box))
                ta = max(1.0, (pb[2] - pb[0]) * (pb[3] - pb[1]))
                ba = max(1.0, (b.box[2] - b.box[0]) * (b.box[3] - b.box[1]))
                size_ok = 1.0 / float(self.cfg.get("max_area_ratio", 4.0)) <= ba / ta <= float(self.cfg.get("max_area_ratio", 4.0))
                if size_ok and (ov >= float(self.cfg["iou_match"]) or d <= maxd):
                    C[i, j] = 1.0 - ov + 0.5 * d / maxd
                else:
                    C[i, j] = 9.0
        matched_t, matched_b = set(), set()
        if C.size:
            r, c = linear_sum_assignment(C)
            for i, j in zip(r, c):
                if C[i, j] < 9.0:
                    t, b = active[i], blobs[j]
                    oc = t.center
                    dt = max(1, frame - t.last_frame)
                    pb = preds[i]
                    t.box = [0.5 * pb[k] + 0.5 * b.box[k] for k in range(4)]
                    nc = ((b.box[0] + b.box[2]) / 2, (b.box[1] + b.box[3]) / 2)
                    v = ((nc[0] - oc[0]) / dt, (nc[1] - oc[1]) / dt)
                    t.vel = (0.7 * t.vel[0] + 0.3 * v[0], 0.7 * t.vel[1] + 0.3 * v[1]) if t.hits > 1 else v
                    t.boxes[frame] = list(b.box)
                    t.hits += 1
                    t.missed = 0
                    t.last_frame = frame
                    t.sources.update(b.sources)
                    t.flows.append(b.flow)
                    if t.hits >= int(self.cfg["min_hits"]):
                        t.confirmed = True
                    matched_t.add(t.id)
                    matched_b.add(j)
        for t in active:
            if t.id not in matched_t:
                t.missed += 1
                if t.missed > int(self.cfg["max_age"]):
                    self.finished.append(self.tracks.pop(t.id))
        for j, b in enumerate(blobs):
            if j in matched_b:
                continue
            t = Track(self._next, frame, frame, list(b.box))
            t.boxes[frame] = list(b.box)
            t.sources.update(b.sources)
            t.flows.append(b.flow)
            self.tracks[t.id] = t
            self._next += 1
        return [t for t in self.tracks.values() if t.missed == 0]

    def is_event(self, t: Track, fps: float) -> bool:
        """A confirmed track that lasted long enough and either got a detector label or really travelled."""
        if not t.confirmed:
            return False
        if (t.last_frame - t.first_frame + 1) / fps < float(self.cfg["min_event_s"]):
            return False
        labeled = t.label()[0] != "unknown"
        return labeled or t.displacement() >= float(self.cfg.get("min_displacement_frac", 0.0)) * self.diag

    def consolidate(self, tracks: List[Track], fps: float) -> List[Track]:
        """Merge fragments of the same object (track broke on a fog patch, gate toggle, or the
        two channels briefly produced separate blobs): B continues A if it starts within a short
        gap after/inside A, close to where A was heading, with compatible labels."""
        gap = int(self.cfg.get("merge_gap_frames", 2 * int(self.cfg["max_age"])))
        maxd = float(self.cfg.get("merge_dist", 0.1)) * self.diag
        ts = sorted([t for t in tracks if t.confirmed], key=lambda t: t.first_frame)
        merged = True
        while merged:
            merged = False
            for a in ts:
                for b in ts:
                    if a is b or b.first_frame < a.first_frame or b.first_frame > a.last_frame + gap:
                        continue
                    la, lb = a.label()[0], b.label()[0]
                    if la != lb and "unknown" not in (la, lb):
                        continue
                    # distance between A (at its frame closest to B's start) and B's first box
                    fa = min(a.boxes, key=lambda f: abs(f - b.first_frame))
                    ba = a.boxes[fa]
                    dtf = b.first_frame - fa
                    ca = ((ba[0] + ba[2]) / 2 + a.vel[0] * dtf, (ba[1] + ba[3]) / 2 + a.vel[1] * dtf)
                    bb = b.boxes[b.first_frame]
                    cb = ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2)
                    near = math.hypot(ca[0] - cb[0], ca[1] - cb[1]) <= maxd
                    if not near:
                        # coexisting fragments of one object (e.g. a long animal split in two blobs)
                        common = set(a.boxes) & set(b.boxes)
                        if common:
                            touch = sum(overlap_small(_grow(a.boxes[f], 0.1), _grow(b.boxes[f], 0.1)) > 0 for f in common)
                            near = touch >= max(1, 0.5 * len(common)) and len(common) <= gap
                    if not near:
                        continue
                    for f, bx in b.boxes.items():
                        if f not in a.boxes or (bx[2] - bx[0]) * (bx[3] - bx[1]) > (a.boxes[f][2] - a.boxes[f][0]) * (a.boxes[f][3] - a.boxes[f][1]):
                            a.boxes[f] = bx
                    a.last_frame = max(a.last_frame, b.last_frame)
                    a.hits += b.hits
                    a.sources.update(b.sources)
                    a.votes.update(b.votes)
                    a.fine_votes.update(b.fine_votes)
                    a.det_hits += b.det_hits
                    a.flows.extend(b.flows)
                    a.vel = b.vel
                    self.merged_into[b.id] = a.id
                    ts.remove(b)
                    merged = True
                    break
                if merged:
                    break
        return ts

    def close(self) -> List[Track]:
        self.finished.extend(self.tracks.values())
        self.tracks = {}
        return self.finished
