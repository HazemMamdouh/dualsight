"""Camera-motion deblurring for the RGB channel.

The gate (``quality.CameraMotionGate``) measures the global inter-frame shift
``(dx, dy)``. During an exposure of ``e`` x frame-interval the sensor sweeps
approximately ``e * (dx, dy)`` pixels, so the blur is modelled as a linear
motion PSF with that length and angle and removed by Wiener deconvolution in
the Fourier domain. The FFTs run in torch, so the same code uses the GPU when
``--device cuda`` is given.

Edges are tapered (blended with a blurred copy) before the FFT to suppress the
ringing a periodic-boundary deconvolution would otherwise create, and the result is
clamped to the local min/max of the input to remove residual ringing around targets
that move differently from the camera.
"""
from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np


def motion_psf(length: float, angle_rad: float, size: int) -> np.ndarray:
    """Normalised linear-motion PSF, anti-aliased, centred in a ``size`` x ``size`` kernel."""
    k = np.zeros((size, size), np.float32)
    c = (size - 1) / 2.0
    L = max(1.0, length)
    n = max(2, int(math.ceil(L * 3)))
    for s in np.linspace(-L / 2, L / 2, n):
        x = c + s * math.cos(angle_rad)
        y = c + s * math.sin(angle_rad)
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        fx, fy = x - x0, y - y0
        for (xx, yy, w) in ((x0, y0, (1 - fx) * (1 - fy)), (x0 + 1, y0, fx * (1 - fy)),
                            (x0, y0 + 1, (1 - fx) * fy), (x0 + 1, y0 + 1, fx * fy)):
            if 0 <= xx < size and 0 <= yy < size:
                k[yy, xx] += w
    return k / max(1e-8, k.sum())


def _taper(img: np.ndarray, psf: np.ndarray, width: int) -> np.ndarray:
    """Blend image borders with a PSF-blurred copy (like MATLAB's edgetaper)."""
    if width <= 0:
        return img
    h, w = img.shape[:2]
    blurred = cv2.filter2D(img, -1, psf, borderType=cv2.BORDER_REFLECT)
    ramp_x = np.minimum(np.arange(w), np.arange(w)[::-1]).astype(np.float32)
    ramp_y = np.minimum(np.arange(h), np.arange(h)[::-1]).astype(np.float32)
    a = np.minimum(np.clip(ramp_x / width, 0, 1)[None, :], np.clip(ramp_y / width, 0, 1)[:, None])
    if img.ndim == 3:
        a = a[..., None]
    return img * a + blurred * (1 - a)


class MotionDeblur:
    def __init__(self, cfg: dict, device: str = "cpu") -> None:
        import torch

        self.torch = torch
        self.cfg = cfg
        self.device = device if device.startswith("cuda") else "cpu"
        self._otf_cache: dict = {}

    def psf_for_shift(self, dx: float, dy: float) -> Tuple[np.ndarray, float, float]:
        L = float(np.hypot(dx, dy)) * float(self.cfg["exposure"])
        L = min(L, float(self.cfg["max_len"]))
        ang = math.atan2(dy, dx)
        size = int(math.ceil(L)) | 1
        size = max(3, size + 2)
        return motion_psf(L, ang, size), L, ang

    def __call__(self, bgr: np.ndarray, dx: float, dy: float) -> np.ndarray:
        if self.cfg.get("method", "wiener") == "off":
            return bgr
        torch = self.torch
        psf, L, _ = self.psf_for_shift(dx, dy)
        if L < 1.5:
            return bgr
        img = bgr.astype(np.float32) / 255.0
        img = _taper(img, psf, int(self.cfg.get("taper", 0)))
        h, w = img.shape[:2]
        # pad PSF to image size with its centre at (0, 0)
        kp = np.zeros((h, w), np.float32)
        ks = psf.shape[0]
        kp[:ks, :ks] = psf
        kp = np.roll(kp, (-(ks // 2), -(ks // 2)), axis=(0, 1))
        dev = self.device
        H = torch.fft.rfft2(torch.from_numpy(kp).to(dev))
        nsr = float(self.cfg["nsr"])
        G = torch.conj(H) / (H.abs() ** 2 + nsr)
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).to(dev)
        X = torch.fft.rfft2(x)
        y = torch.fft.irfft2(X * G[None], s=(h, w))
        out = np.ascontiguousarray(y.clamp(0, 1).mul(255.0).round().byte().permute(1, 2, 0).cpu().numpy())
        k = int(self.cfg.get("dering", 0))
        if k > 1:
            # deconvolution can only *redistribute* intensity along the blur path; anything outside the
            # local range of the blurred input is ringing (PSF error, objects moving against the camera)
            ker = np.ones((k, k), np.uint8)
            out = np.clip(out, cv2.erode(bgr, ker), cv2.dilate(bgr, ker))
        return out
