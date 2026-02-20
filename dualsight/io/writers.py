"""Video writer: H.264 through an FFmpeg pipe when available (browser/GitHub friendly),
OpenCV ``mp4v`` otherwise."""
from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("dualsight")


class VideoWriter:
    def __init__(self, path: str, fps: float, size: Optional[Tuple[int, int]] = None, crf: int = 23) -> None:
        self.path, self.fps, self.size, self.crf = path, fps, size, crf
        self._proc = None
        self._cv = None

    def _open(self, w: int, h: int) -> None:
        self.size = (w, h)
        if shutil.which("ffmpeg"):
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                   "-s", f"{w}x{h}", "-r", f"{self.fps:.4f}", "-i", "-", "-an", "-c:v", "libx264",
                   "-preset", "veryfast", "-crf", str(self.crf), "-pix_fmt", "yuv420p",
                   "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-movflags", "+faststart", self.path]
            try:
                self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                return
            except OSError:
                self._proc = None
        self._cv = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        if not self._cv.isOpened():
            raise RuntimeError(f"cannot open video writer: {self.path}")

    def write(self, frame: np.ndarray) -> None:
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        h, w = frame.shape[:2]
        if self._proc is None and self._cv is None:
            self._open(w, h)
        if (w, h) != self.size:
            frame = cv2.resize(frame, self.size)
        if self._proc is not None:
            self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            self._cv.write(frame)

    def release(self) -> None:
        if self._proc is not None:
            self._proc.stdin.close()
            self._proc.wait()
            self._proc = None
        if self._cv is not None:
            self._cv.release()
            self._cv = None
