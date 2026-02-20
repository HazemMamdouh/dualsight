"""Per-channel moving-object detection.

Two complementary detectors run on every channel and are fused:

1. **Ego-compensated dense optical flow.** A similarity transform fitted with
   RANSAC to a grid of flow vectors models the camera motion; subtracting it
   leaves the *residual* flow produced by independently moving objects. The
   residual magnitude is thresholded adaptively (``max(abs, median + k*MAD)``),
   so fog drift and sensor noise raise the bar automatically.

2. **Background model / aligned differencing.** While the camera is still,
   MOG2 (or KNN) yields full silhouettes. As soon as the camera moves the
   background model is invalid, so the detector switches to differencing
   against the previous frame *warped by the estimated camera motion*, and
   re-learns the background once the rig settles.

Fusion (``combine``): ``and`` keeps flow regions confirmed by the background
cue (fewest false alarms - the default for perimeter use), ``or`` maximises
recall.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from .flow import FlowEstimator


@dataclass
class Blob:
    box: List[float]          # x1, y1, x2, y2 in the channel's processing coordinates
    area: float
    flow: List[float]         # mean residual flow (dx, dy) px/frame
    magnitude: float


@dataclass
class ChannelMotion:
    mask: np.ndarray                      # uint8 0/255, processing resolution
    residual_mag: np.ndarray              # float32
    camera: np.ndarray                    # 2x3 affine prev->cur
    camera_shift: float                   # px/frame at processing resolution
    score: float                          # fraction of moving pixels
    blobs: List[Blob] = field(default_factory=list)
    flow_thresh: float = 0.0


def estimate_camera(flow: np.ndarray, step: int, ransac_px: float, gray: Optional[np.ndarray] = None,
                    min_texture: float = 6.0) -> np.ndarray:
    """Similarity transform of the camera motion, fitted on *textured* grid points only:
    in flat regions (sky, dense fog) flow is unobservable and reads ~0, which would
    otherwise masquerade as a still camera."""
    h, w = flow.shape[:2]
    ys, xs = np.mgrid[step // 2:h:step, step // 2:w:step]
    ys, xs = ys.ravel(), xs.ravel()
    if gray is not None:
        g = cv2.GaussianBlur(gray, (0, 0), 1.5).astype(np.float32)
        tex = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
        keep = tex[ys, xs] > min_texture
        if keep.sum() >= 30:
            ys, xs = ys[keep], xs[keep]
    p0 = np.stack([xs, ys], 1).astype(np.float32)
    f = flow[ys, xs]
    mag = np.linalg.norm(f, axis=1)
    if (mag < 0.35).mean() > 0.4:  # most of the scene is still -> static camera
        return np.float32([[1, 0, 0], [0, 1, 0]])
    M, inl = cv2.estimateAffinePartial2D(p0, p0 + f, method=cv2.RANSAC, ransacReprojThreshold=ransac_px, maxIters=300)
    if M is None:
        return np.float32([[1, 0, 0], [0, 1, 0]])
    return M.astype(np.float32)


def camera_flow(M: np.ndarray, h: int, w: int) -> np.ndarray:
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    fx = M[0, 0] * xs + M[0, 1] * ys + M[0, 2] - xs
    fy = M[1, 0] * xs + M[1, 1] * ys + M[1, 2] - ys
    return np.dstack([fx, fy])


class MotionDetector:
    def __init__(self, name: str, cfg: dict, flow: FlowEstimator) -> None:
        self.name = name
        self.cfg = cfg
        self.flow = flow
        self.prev: Optional[np.ndarray] = None
        bg = cfg["bg"]
        if bg["method"] == "knn":
            self.bg = cv2.createBackgroundSubtractorKNN(history=bg["history"], detectShadows=False)
        else:
            self.bg = cv2.createBackgroundSubtractorMOG2(history=bg["history"], varThreshold=bg["var_threshold"],
                                                         detectShadows=False)
        self.use_bg_model = bg["method"] in ("mog2", "knn")
        self.static_frames = 0
        k = int(cfg["morph"])
        self.k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self.k_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3 * k, 3 * k))
        self.k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def reset(self) -> None:
        self.prev = None
        self.static_frames = 0

    def __call__(self, gray: np.ndarray, camera_hint: Optional[np.ndarray] = None,
                 snr_boost: float = 1.0) -> ChannelMotion:
        """``camera_hint``: 2x3 prev->cur camera transform in this channel's processing pixels,
        shared from a more reliable estimator (both sensors sit on the same rig)."""
        h, w = gray.shape
        ident = np.float32([[1, 0, 0], [0, 1, 0]])
        if self.prev is None or self.prev.shape != gray.shape:
            self.prev = gray
            if self.use_bg_model:
                self.bg.apply(gray, learningRate=1.0)
            z = np.zeros((h, w), np.uint8)
            return ChannelMotion(z, np.zeros((h, w), np.float32), ident, 0.0, 0.0)

        flow = self.flow(self.prev, gray)
        ego = self.cfg["ego"]
        if not ego["enabled"]:
            M = ident
        elif camera_hint is not None:
            M = camera_hint.astype(np.float32)
        else:
            M = estimate_camera(flow, ego["grid_step"], ego["ransac_px"], gray)
        shift = float(np.hypot(M[0, 2], M[1, 2]))
        cam_moving = not np.allclose(M, ident, atol=1e-3) and shift > 0.25
        resid = flow - camera_flow(M, h, w) if cam_moving else flow
        mag = cv2.magnitude(resid[..., 0], resid[..., 1])

        med = float(np.median(mag))
        mad = float(np.median(np.abs(mag - med))) + 1e-6
        thr = max(float(self.cfg["flow_thresh"]), med + float(self.cfg["flow_k_mad"]) * 1.4826 * mad)
        if cam_moving:
            # the ego-motion model is only approximate (parallax, rolling shutter, blur), so the
            # residual that counts as independent motion grows with the camera speed
            thr += float(self.cfg["ego"].get("resid_k", 0.0)) * shift
        flow_mask = (mag > thr).astype(np.uint8) * 255
        # exclude the border strip that becomes invalid under camera motion
        if cam_moving:
            b = int(min(max(abs(M[0, 2]), abs(M[1, 2])) + 4, min(h, w) // 4))
            flow_mask[:b] = 0
            flow_mask[-b:] = 0
            flow_mask[:, :b] = 0
            flow_mask[:, -b:] = 0

        # --- background cue
        bgc = self.cfg["bg"]
        if cam_moving:
            self.static_frames = 0
        else:
            self.static_frames += 1
        use_model = self.use_bg_model and self.static_frames > int(bgc["warmup_frames"])
        if self.use_bg_model:
            lr = -1 if self.static_frames > int(bgc["warmup_frames"]) else 0.5
            fg = self.bg.apply(gray, learningRate=lr)
        if use_model:
            bg_mask = (fg > 200).astype(np.uint8) * 255
        else:
            warped = cv2.warpAffine(self.prev, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            diff = cv2.absdiff(cv2.GaussianBlur(gray, (5, 5), 0), cv2.GaussianBlur(warped, (5, 5), 0))
            bg_mask = (diff > int(bgc["diff_thresh"])).astype(np.uint8) * 255

        # the background cue must be *solid*: isolated speckle (noise, codec flicker,
        # amplified fog texture) is removed before it is allowed to confirm flow
        bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, self.k_open)
        if self.cfg["combine"] == "or":
            mask = cv2.bitwise_or(flow_mask, bg_mask)
        else:
            mask = cv2.bitwise_and(flow_mask, cv2.dilate(bg_mask, self.k_big))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.k_small)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.k_big)

        # --- blobs
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        min_area = float(self.cfg["min_area_frac"]) * h * w
        max_area = float(self.cfg.get("max_area_frac", 1.0)) * h * w
        max_w = float(self.cfg.get("max_width_frac", 1.0)) * w
        blobs: List[Blob] = []
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area < min_area or area > max_area or bw > max_w:
                mask[lab == i] = 0
                continue
            sel = lab[y:y + bh, x:x + bw] == i
            r = resid[y:y + bh, x:x + bw][sel]
            mf = r.mean(axis=0)
            mmag = float(mag[y:y + bh, x:x + bw][sel].mean())
            if mmag < float(self.cfg.get("min_blob_snr", 0.0)) * snr_boost * thr:
                mask[lab == i] = 0  # barely above the noise floor on average -> drifting haze, shimmer
                continue
            blobs.append(Blob([float(x), float(y), float(x + bw), float(y + bh)], float(area),
                              [float(mf[0]), float(mf[1])], mmag))
        self.prev = gray
        score = float((mask > 0).mean())
        return ChannelMotion(mask, mag, M, shift, score, blobs, thr)
