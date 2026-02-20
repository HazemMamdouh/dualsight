"""Thermal-channel enhancement.

Why not *only* ESRGAN?
    Fog in LWIR is mostly a low-frequency radiance veil: ``I = J*t + A*(1-t)``.
    A super-resolution GAN restores fine texture but does not undo the veil, so
    the chain here is

        classical  :  destripe -> bilateral -> veil-aware dehaze -> CLAHE -> unsharp
        esrgan     :  destripe -> Real-ESRGAN x4 (published weights) -> area-downsample back
        hybrid     :  destripe -> Real-ESRGAN x4 (acts as a learned denoiser/detail
                      restorer) -> veil-aware dehaze -> CLAHE -> unsharp

    Running the GAN *before* the veil inversion matters: ``1/t`` amplifies whatever
    noise is left, so it should see the cleanest possible signal.

    ``classical`` runs in a few ms on a CPU; ``hybrid`` is the recommended GPU mode.
    The dehaze strength is modulated by the fog score from the gate, so light haze
    gets a light touch.
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import cv2
import numpy as np

from ..utils.device import is_gpu

log = logging.getLogger("dualsight")


def guided_filter(I: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """He et al. guided filter (box-filter formulation), float32 inputs in [0, 1]."""
    k = (2 * r + 1, 2 * r + 1)
    mean_I = cv2.boxFilter(I, -1, k)
    mean_p = cv2.boxFilter(p, -1, k)
    cov_Ip = cv2.boxFilter(I * p, -1, k) - mean_I * mean_p
    var_I = cv2.boxFilter(I * I, -1, k) - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    return cv2.boxFilter(a, -1, k) * I + cv2.boxFilter(b, -1, k)


def destripe(gray: np.ndarray, sigma: float = 3.0, max_amp: float = 3.0) -> np.ndarray:
    """Remove residual column fixed-pattern noise typical of uncorrected microbolometers.

    Stripes are column-constant offsets, so they survive in ``I - horizontal_blur(I)`` with the
    same value on every row; scene edges do not. The robust (median) column profile of that
    residual is the stripe estimate. Only enable it for cores without a working NUC."""
    I = gray.astype(np.float32)
    resid = I - cv2.GaussianBlur(I, (0, 0), sigmaX=sigma, sigmaY=0.1)
    prof = np.clip(np.median(resid, axis=0), -max_amp, max_amp)
    return np.clip(I - prof[None, :], 0, 255).astype(np.uint8)


class TemporalNR:
    """Motion-compensated, motion-adaptive recursive filter (standard in thermal cores).

    The accumulator is first re-aligned to the current frame with a phase-correlation
    shift, so a panning/shaking rig does not smear the background. Static pixels are
    then averaged over time (sensor noise and codec flicker vanish, which matters
    because the veil inversion amplifies them by ``1/t``); pixels that change by more
    than the noise band follow the input immediately, so moving targets are not smeared."""

    def __init__(self, alpha_min: float = 0.25, lo: float = 4.0, hi: float = 14.0) -> None:
        self.alpha_min, self.lo, self.hi = alpha_min, lo, hi
        self.acc: Optional[np.ndarray] = None
        self._win: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.acc = None

    def __call__(self, gray: np.ndarray, shift: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """``shift``: known global displacement prev->cur (e.g. the rig ego-motion measured on
        the RGB channel). Without it, phase correlation on the frame itself is used."""
        x = gray.astype(np.float32)
        if self.acc is None or self.acc.shape != x.shape:
            self.acc = x
            return gray
        if shift is not None:
            dx, dy = shift
            resp = 1.0
        else:
            if self._win is None or self._win.shape != x.shape:
                self._win = cv2.createHanningWindow(x.shape[::-1], cv2.CV_32F)
            (dx, dy), resp = cv2.phaseCorrelate(self.acc.copy(), x.copy(), self._win)
        if resp < 0.05 or abs(dx) > x.shape[1] / 8 or abs(dy) > x.shape[0] / 8:
            self.acc = x  # cannot register -> restart the recursion
            return gray
        if abs(dx) > 0.2 or abs(dy) > 0.2:
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            self.acc = cv2.warpAffine(self.acc, M, (x.shape[1], x.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        d = cv2.GaussianBlur(np.abs(x - self.acc), (5, 5), 0)
        a = np.clip((d - self.lo) / (self.hi - self.lo), 0.0, 1.0) * (1 - self.alpha_min) + self.alpha_min
        self.acc = a * x + (1 - a) * self.acc
        return np.clip(self.acc + 0.5, 0, 255).astype(np.uint8)


def _ema(state: Optional[dict], key: str, value: float, beta: float = 0.85) -> float:
    if state is None:
        return value
    v = state.get(key)
    v = value if v is None else beta * v + (1 - beta) * value
    state[key] = v
    return v


def dehaze_thermal(gray: np.ndarray, strength: float, cfg: dict, prefiltered: bool = False,
                   state: Optional[dict] = None) -> np.ndarray:
    """Single-channel veil removal adapted from the dark-channel prior.

    In white-hot LWIR the fog radiance ``A`` sits between cold sky and warm
    targets, so instead of a dark channel we use the local *distance to A*:
    clear pixels deviate strongly from ``A``; veiled ones collapse towards it.
    """
    g = gray
    if cfg.get("denoise", 0) and not prefiltered:
        # denoise *before* inverting the veil, otherwise 1/t amplifies sensor noise
        d = int(cfg["denoise"])
        g = cv2.bilateralFilter(g, d * 2 + 1, 12, d * 2)
    I = g.astype(np.float32) / 255.0
    low = cv2.GaussianBlur(I, (0, 0), sigmaX=max(8, I.shape[1] / 40))
    # global parameters are smoothed over time (``state``) so the output does not flicker
    A = _ema(state, "A", float(np.median(low)))                # veil radiance
    dev = np.abs(I - A)
    k = int(cfg["window"]) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    # local contrast envelope: max deviation from A in a window
    env = cv2.dilate(dev, kernel)
    env = cv2.GaussianBlur(env, (0, 0), k / 3)
    ref = _ema(state, "ref", max(1e-3, float(np.percentile(env, 99))))
    t = np.clip(env / ref, 0.0, 1.0)
    omega = float(cfg["omega"]) * float(np.clip(strength, 0.0, 1.0))
    t = 1.0 - omega * (1.0 - t)
    t = guided_filter(I, t, int(cfg["guided_radius"]), float(cfg["guided_eps"]))
    t = np.maximum(t, float(cfg["t_min"]))
    J = (I - A) / t + A
    # robust stretch to full range
    lo, hi = np.percentile(J, (0.5, 99.7))
    lo, hi = _ema(state, "lo", float(lo)), _ema(state, "hi", float(hi))
    J = np.clip((J - lo) / max(1e-6, hi - lo), 0, 1)
    out = (J * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=float(cfg["clahe_clip"]), tileGridSize=(cfg["clahe_grid"],) * 2)
    out = clahe.apply(out)
    amt = float(cfg.get("unsharp", 0.0))
    if amt > 0:
        blur = cv2.GaussianBlur(out, (0, 0), 1.6)
        out = cv2.addWeighted(out, 1 + amt, blur, -amt, 0)
    return out


class RealESRGAN:
    def __init__(self, cfg: dict, device: str) -> None:
        import torch

        from .esrgan_arch import MODELS

        name = cfg["model"]
        if name not in MODELS:
            raise ValueError(f"unknown ESRGAN model {name}; choose from {list(MODELS)}")
        builder, url = MODELS[name]
        wdir = os.path.expanduser(cfg["weights_dir"])
        os.makedirs(wdir, exist_ok=True)
        path = os.path.join(wdir, os.path.basename(url))
        if not os.path.exists(path):
            log.info("downloading %s -> %s", url, path)
            torch.hub.download_url_to_file(url, path, progress=False)
        sd = torch.load(path, map_location="cpu", weights_only=True)
        sd = sd.get("params_ema", sd.get("params", sd))
        net = builder()
        net.load_state_dict(sd, strict=True)
        net.eval()
        self.torch = torch
        self.device = device
        self.half = bool(cfg.get("half")) and device.startswith("cuda")
        self.net = net.to(device).half() if self.half else net.to(device)
        self.tile = int(cfg.get("tile") or 0)
        self.keep = bool(cfg.get("keep_sr_resolution"))

    def _run(self, x):
        with self.torch.no_grad():
            return self.net(x)

    def __call__(self, gray: np.ndarray) -> np.ndarray:
        torch = self.torch
        h, w = gray.shape[:2]
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB) if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(img).permute(2, 0, 1)[None].float().div_(255.0).to(self.device)
        if self.half:
            x = x.half()
        if self.tile <= 0:
            y = self._run(x)
        else:
            y = torch.zeros((1, 3, h * 4, w * 4), dtype=x.dtype, device=x.device)
            T, pad = self.tile, 10
            for y0 in range(0, h, T):
                for x0 in range(0, w, T):
                    ys, xs = max(0, y0 - pad), max(0, x0 - pad)
                    ye, xe = min(h, y0 + T + pad), min(w, x0 + T + pad)
                    out = self._run(x[..., ys:ye, xs:xe])
                    oy, ox = (y0 - ys) * 4, (x0 - xs) * 4
                    th, tw = min(T, h - y0) * 4, min(T, w - x0) * 4
                    y[..., y0 * 4:y0 * 4 + th, x0 * 4:x0 * 4 + tw] = out[..., oy:oy + th, ox:ox + tw]
        y = y.float().clamp(0, 1).mul(255.0).round().byte()[0].permute(1, 2, 0).cpu().numpy()
        y = cv2.cvtColor(y, cv2.COLOR_RGB2GRAY)
        if not self.keep:
            y = cv2.resize(y, (w, h), interpolation=cv2.INTER_AREA)
        return y


class ThermalEnhancer:
    def __init__(self, cfg: dict, device: str) -> None:
        method = cfg["enhancer"]
        if method == "auto":
            method = "hybrid" if is_gpu(device) else "classical"
        self.method = method
        self.cfg = cfg
        tn = float(cfg["dehaze"].get("temporal_nr", 0.25) or 0)
        self.use_tnr = tn > 0
        self.tnr = TemporalNR(alpha_min=tn if tn > 0 else 1.0)
        self.state: dict = {}
        self.sr: Optional[RealESRGAN] = None
        if method in ("esrgan", "hybrid"):
            try:
                self.sr = RealESRGAN(cfg["esrgan"], device)
            except Exception as e:  # pragma: no cover - network / weights problems
                log.warning("Real-ESRGAN unavailable (%s) -> falling back to classical dehaze", e)
                self.method = "classical"
        log.info("thermal enhancer: %s", self.method)

    def reset(self) -> None:
        """Call when enhancement is (re)activated so temporal state starts fresh."""
        self.tnr.reset()
        self.state = {}

    def __call__(self, frame: np.ndarray, fog_score: float, shift: Optional[Tuple[float, float]] = None) -> np.ndarray:
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.method == "off":
            return gray
        strength = float(np.clip(fog_score * 1.6, 0.35, 1.0))
        g = destripe(gray) if self.cfg["dehaze"].get("destripe", False) else gray
        if self.use_tnr:
            g = self.tnr(g, shift)
        if self.method == "classical" or self.sr is None:
            return dehaze_thermal(g, strength, self.cfg["dehaze"], prefiltered=False, state=self.state)
        g = self.sr(g)
        if self.method == "hybrid":
            g = dehaze_thermal(g, strength, self.cfg["dehaze"], prefiltered=True, state=self.state)
        return g
