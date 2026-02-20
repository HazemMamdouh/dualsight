"""Device resolution and capability probing (CUDA via torch and/or OpenCV-CUDA)."""
from __future__ import annotations

import functools
import logging

import cv2

log = logging.getLogger("dualsight")


@functools.lru_cache(maxsize=None)
def torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover
        return False


@functools.lru_cache(maxsize=None)
def opencv_cuda_available() -> bool:
    try:
        return hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False


@functools.lru_cache(maxsize=None)
def opencv_has_gstreamer() -> bool:
    info = cv2.getBuildInformation()
    for line in info.splitlines():
        if "GStreamer" in line:
            return "YES" in line
    return False


def resolve_device(device: str) -> str:
    """Map ``auto`` / ``cuda`` / ``cuda:N`` / ``cpu`` to a usable torch device string."""
    device = (device or "auto").lower()
    if device == "auto":
        return "cuda:0" if torch_cuda_available() else "cpu"
    if device.startswith("cuda") or device.isdigit():
        if not torch_cuda_available():
            log.warning("CUDA requested but torch.cuda is not available -> falling back to CPU")
            return "cpu"
        return device if device.startswith("cuda") else f"cuda:{device}"
    if device == "mps":
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"
    return "cpu"


def is_gpu(device: str) -> bool:
    return device.startswith("cuda") or device == "mps"
