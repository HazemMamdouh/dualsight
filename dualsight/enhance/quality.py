"""Classical, model-free checks that decide *whether* each channel needs enhancement.

Thermal channel - fog / low-contrast check
    Fog (and haze, smoke, drizzle) adds a nearly uniform radiance veil, which
    compresses the dynamic range and kills high-frequency energy. We measure

    * ``spread``  - robust intensity range (p99 - p1) / 255
    * ``grad``    - mean Sobel gradient magnitude (edge energy)
    * ``entropy`` - histogram entropy (bits)

    and combine the first two into a ``fog_score`` in [0, 1] (1 = fully veiled).
    A hysteresis gate turns enhancement on/off so it does not flicker.

RGB channel - camera-motion check
    Global inter-frame motion is estimated from grid-distributed Shi-Tomasi
    corners tracked with pyramidal Lucas-Kanade and a RANSAC similarity fit.
    Distributing the corners over a grid keeps a large, textured moving target
    from out-voting the background. Phase correlation is the fallback when too
    few corners survive (e.g. dense fog).
    The shift magnitude drives the gate, and its direction/length parameterise
    the motion-blur PSF used by the deblurring stage. Optionally a sharpness
    drop (variance of Laplacian vs. a running baseline) must also be present.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


def _to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def fog_metrics(img: np.ndarray, spread_ref: float = 0.55, grad_ref: float = 9.0) -> dict:
    g = _to_gray(img)
    if g.shape[1] > 640:
        g = cv2.resize(g, (640, int(640 * g.shape[0] / g.shape[1])), interpolation=cv2.INTER_AREA)
    p1, p99 = np.percentile(g, (1, 99))
    spread = float(p99 - p1) / 255.0
    # measure *structure*, not sensor noise: half-res + blur before the gradient
    gs = cv2.GaussianBlur(cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2), interpolation=cv2.INTER_AREA), (0, 0), 1.2)
    gx = cv2.Sobel(gs, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gs, cv2.CV_32F, 0, 1, ksize=3)
    grad = float(np.mean(cv2.magnitude(gx, gy)))
    hist = np.bincount(g.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    s_spread = 1.0 - min(1.0, spread / spread_ref)
    s_grad = 1.0 - min(1.0, grad / grad_ref)
    fog = float(np.clip(0.5 * s_spread + 0.5 * s_grad, 0.0, 1.0))
    return {"spread": spread, "grad": grad, "entropy": entropy, "fog_score": fog}


class HysteresisGate:
    def __init__(self, on: float, off: float, hold: int) -> None:
        self.on, self.off, self.hold = on, off, max(1, hold)
        self.state = False
        self._cnt = 0

    def update(self, value: float) -> bool:
        want = value >= self.on if not self.state else value > self.off
        if want != self.state:
            self._cnt += 1
            if self._cnt >= self.hold:
                self.state, self._cnt = want, 0
        else:
            self._cnt = 0
        return self.state


class FogGate:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.gate = HysteresisGate(cfg["on_score"], cfg["off_score"], cfg["hold_frames"])
        self.ema: Optional[float] = None

    def __call__(self, img: np.ndarray) -> Tuple[bool, dict]:
        m = fog_metrics(img, self.cfg["spread_ref"], self.cfg["grad_ref"])
        self.ema = m["fog_score"] if self.ema is None else 0.7 * self.ema + 0.3 * m["fog_score"]
        m["fog_score_smooth"] = self.ema
        force = self.cfg.get("force")
        state = self.gate.update(self.ema) if force is None else bool(force)
        return state, m


@dataclass
class CameraMotion:
    dx: float          # full-resolution pixels / frame
    dy: float
    response: float    # inlier / static-support ratio (LK) or phase-correlation peak
    sharpness: float
    blur_ratio: float
    reliable: bool = True  # True when estimated from tracked corners (shared as the rig ego-motion)

    @property
    def magnitude(self) -> float:
        return float(np.hypot(self.dx, self.dy))


def grid_corners(gray: np.ndarray, cols: int = 8, rows: int = 6, per_cell: int = 8) -> np.ndarray:
    """Shi-Tomasi corners, capped per grid cell so no single region dominates."""
    h, w = gray.shape
    pts = []
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * h // rows, (r + 1) * h // rows
            x0, x1 = c * w // cols, (c + 1) * w // cols
            p = cv2.goodFeaturesToTrack(gray[y0:y1, x0:x1], per_cell, 0.005, 6, blockSize=5)
            if p is not None:
                pts.append(p.reshape(-1, 2) + np.float32([x0, y0]))
    return np.concatenate(pts).astype(np.float32) if pts else np.zeros((0, 2), np.float32)


def robust_global_shift(prev: np.ndarray, cur: np.ndarray, min_inliers: int = 25) -> Optional[Tuple[float, float, float]]:
    """(dx, dy, inlier_ratio) of the dominant (background) motion between two gray frames."""
    p0 = grid_corners(prev)
    if len(p0) < min_inliers:
        return None
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0.reshape(-1, 1, 2), None, winSize=(21, 21), maxLevel=4)
    ok = st.ravel() == 1
    if ok.sum() < min_inliers:
        return None
    a, b = p0[ok], p1.reshape(-1, 2)[ok]
    # A still camera leaves most of the background static. Moving targets can
    # dominate a RANSAC vote when the background is weakly textured, so a large
    # static support short-circuits the estimate to "no camera motion".
    static = np.linalg.norm(b - a, axis=1) < 0.6
    if static.mean() > 0.35:
        return 0.0, 0.0, float(static.mean())
    M, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=1.0, maxIters=500)
    if M is None or inl is None or inl.sum() < min_inliers:
        return None
    inl = inl.ravel() == 1
    d = np.median(b[inl] - a[inl], axis=0)
    return float(d[0]), float(d[1]), float(inl.mean())


class CameraMotionGate:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.prev: Optional[np.ndarray] = None
        self.win: Optional[np.ndarray] = None
        self.sharp_hist: deque = deque(maxlen=90)
        self.gate = HysteresisGate(1.0, 0.5, cfg["hold_frames"])
        self.last = CameraMotion(0, 0, 0, 0, 1)

    def __call__(self, img: np.ndarray) -> Tuple[bool, CameraMotion]:
        g = _to_gray(img)
        aw = int(self.cfg["analysis_width"])
        scale = g.shape[1] / aw
        small = cv2.resize(g, (aw, int(round(g.shape[0] / scale))), interpolation=cv2.INTER_AREA).astype(np.float32)
        if self.win is None or self.win.shape != small.shape:
            self.win = cv2.createHanningWindow(small.shape[::-1], cv2.CV_32F)
        sharp = float(cv2.Laplacian(small, cv2.CV_32F).var())
        if self.prev is None:
            self.prev = small
            self.sharp_hist.append(sharp)
            return False, self.last
        est = robust_global_shift(self.prev.astype(np.uint8), small.astype(np.uint8))
        reliable = est is not None
        if est is not None:
            dx, dy, resp = est
        else:
            # copies: some OpenCV builds window the inputs of phaseCorrelate in place
            (dx, dy), resp = cv2.phaseCorrelate(self.prev.copy(), small.copy(), self.win)
            if resp < 0.03:  # unreliable peak (e.g. fully fogged frame) -> assume static
                dx = dy = 0.0
        self.prev = small
        dx, dy = dx * scale, dy * scale
        base = float(np.median(self.sharp_hist)) if self.sharp_hist else sharp
        ratio = sharp / max(1e-6, base)
        mag = float(np.hypot(dx, dy))
        if mag < self.cfg["shift_px"]:
            self.sharp_hist.append(sharp)  # baseline is learnt only while the camera is still
        moving = mag >= self.cfg["shift_px"]
        if self.cfg.get("require_blur"):
            moving = moving and ratio < self.cfg["blur_ratio"]
        state = self.gate.update(1.0 if moving else 0.0)
        force = self.cfg.get("force")
        if force is not None:
            state = bool(force)
        self.last = CameraMotion(float(dx), float(dy), float(resp), sharp, float(ratio), reliable)
        return state, self.last
