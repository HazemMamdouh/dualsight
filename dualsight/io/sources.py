"""Synchronised dual-stream ingestion.

Each channel is opened through a GStreamer pipeline (when OpenCV is built with
GStreamer) or plain OpenCV/FFmpeg as a fallback. Offline files, RTSP/UDP feeds,
V4L2 cameras and raw ``gst:`` pipelines are all accepted, so the same code path
serves bench replays and live deployment.

Pairing is done on presentation timestamps: the thermal channel is the master
clock and, for every thermal frame, the RGB frame with the closest timestamp is
selected (frames that drift beyond ``sync_tolerance_ms`` are reported). With
``realtime=True`` the reader paces offline files at their native rate and drops
frames whenever processing falls behind - exactly what a live feed does.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

from ..utils.device import opencv_has_gstreamer

log = logging.getLogger("dualsight")


# --------------------------------------------------------------------------------------
# GStreamer pipeline construction
# --------------------------------------------------------------------------------------

_DECODERS = {
    "none": "decodebin",
    "nvidia": "nvh264dec",                        # desktop NVDEC (gst-plugins-bad)
    "jetson": "nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx",
}

_SINK = "videoconvert ! video/x-raw,format=BGR ! appsink drop={drop} max-buffers={maxbuf} sync={sync}"


def build_gst_pipeline(spec: str, hw_decode: str = "none", live_pacing: bool = False) -> str:
    """Return a GStreamer launch string ending in an OpenCV-compatible appsink."""
    sync = "true" if live_pacing else "false"
    drop = "true" if live_pacing else "false"
    sink = _SINK.format(drop=drop, maxbuf=1 if live_pacing else 8, sync=sync)
    dec = _DECODERS.get(hw_decode, "decodebin")

    if spec.startswith("gst:"):
        return spec[4:]
    if spec.startswith("rtsp://"):
        dec_chain = "decodebin" if hw_decode == "none" else f"rtph264depay ! h264parse ! {dec}"
        src = f'rtspsrc location="{spec}" latency=100 ! {dec_chain}'
        return f"{src} ! {sink}"
    if spec.startswith("udp://"):
        port = spec.rsplit(":", 1)[-1]
        caps = "application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96"
        d = "avdec_h264" if hw_decode == "none" else dec
        return f'udpsrc port={port} caps="{caps}" ! rtpjitterbuffer ! rtph264depay ! h264parse ! {d} ! {sink}'
    if spec.startswith("/dev/video") or spec.isdigit():
        dev = spec if spec.startswith("/dev/") else f"/dev/video{spec}"
        return f"v4l2src device={dev} ! {sink}"
    # local file
    if hw_decode == "none":
        return f'filesrc location="{spec}" ! decodebin ! {sink}'
    return f'filesrc location="{spec}" ! qtdemux ! h264parse ! {dec} ! {sink}'


# --------------------------------------------------------------------------------------
# single stream
# --------------------------------------------------------------------------------------


class FrameSource:
    """One video channel with timestamps."""

    def __init__(self, spec: str, name: str, use_gstreamer: str | bool = "auto", hw_decode: str = "none",
                 live_pacing: bool = False) -> None:
        self.spec = str(spec)
        self.name = name
        self.backend = "opencv"
        want_gst = (use_gstreamer is True or str(use_gstreamer).lower() == "true" or
                    (str(use_gstreamer).lower() == "auto" and opencv_has_gstreamer()))
        if want_gst and not opencv_has_gstreamer():
            log.warning("[%s] GStreamer requested but OpenCV was built without it -> using FFmpeg backend", name)
            want_gst = False

        self.cap = None
        if want_gst:
            pipe = build_gst_pipeline(self.spec, hw_decode, live_pacing)
            log.info("[%s] GStreamer: %s", name, pipe)
            self.cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
            if self.cap.isOpened():
                self.backend = "gstreamer"
            else:
                log.warning("[%s] GStreamer pipeline failed to open -> falling back to OpenCV", name)
                self.cap = None
        if self.cap is None:
            src = int(self.spec) if self.spec.isdigit() else self.spec
            self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"[{name}] cannot open source: {spec}")

        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.frame_count = n if n > 0 else None
        self.is_file = os.path.isfile(self.spec)
        self._idx = -1
        self._last_ts = -1.0

    def _timestamp(self) -> float:
        ts = float(self.cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        synth = self._idx * 1000.0 / self.fps
        # some backends report 0 or non-monotonic stamps -> fall back to the frame clock
        if ts <= self._last_ts or (self._idx > 0 and ts == 0.0):
            ts = synth
        self._last_ts = ts
        return ts

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        ok, frame = self.cap.read()
        if not ok:
            return False, None, -1.0
        self._idx += 1
        return True, frame, self._timestamp()

    def grab(self) -> Tuple[bool, float]:
        ok = self.cap.grab()
        if not ok:
            return False, -1.0
        self._idx += 1
        return True, self._timestamp()

    def seek_seconds(self, s: float) -> None:
        if s > 0 and self.is_file:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, s * 1000.0)
            self._idx = int(round(s * self.fps)) - 1
            self._last_ts = s * 1000.0 - 1e-3

    def release(self) -> None:
        self.cap.release()


# --------------------------------------------------------------------------------------
# synchronised pair
# --------------------------------------------------------------------------------------


@dataclass
class FramePair:
    index: int          # output index (0..)
    t: float            # seconds on the master (thermal) clock
    thermal: np.ndarray
    rgb: np.ndarray
    skew_ms: float      # |t_thermal - t_rgb|
    dropped: int        # master frames dropped before this one (realtime mode)


class SyncedDualSource:
    def __init__(self, thermal_spec: str, rgb_spec: str, cfg: dict) -> None:
        sc = cfg["sources"]
        live = bool(sc.get("realtime"))
        self.thermal = FrameSource(thermal_spec, "thermal", sc["use_gstreamer"], sc["hw_decode"], live_pacing=False)
        self.rgb = FrameSource(rgb_spec, "rgb", sc["use_gstreamer"], sc["hw_decode"], live_pacing=False)
        self.tol = float(sc["sync_tolerance_ms"])
        self.realtime = live
        self.max_frames = int(sc.get("max_frames") or 0)
        start = float(sc.get("start_s") or 0.0)
        self.thermal.seek_seconds(start)
        self.rgb.seek_seconds(start)
        self.fps = self.thermal.fps
        self.n_skew_violations = 0
        self.n_dropped = 0
        n = self.thermal.frame_count
        if n is not None and start > 0:
            n = max(0, n - int(round(start * self.fps)))
        if self.max_frames and n:
            n = min(n, self.max_frames)
        self.expected_frames = n
        log.info("thermal: %dx%d @ %.2f fps (%s) | rgb: %dx%d @ %.2f fps (%s)",
                 self.thermal.width, self.thermal.height, self.thermal.fps, self.thermal.backend,
                 self.rgb.width, self.rgb.height, self.rgb.fps, self.rgb.backend)
        if abs(self.thermal.fps - self.rgb.fps) > 0.5:
            log.info("channels run at different rates -> RGB is resampled onto the thermal clock")

    def __iter__(self) -> Iterator[FramePair]:
        rgb_prev = None  # (frame, ts)
        rgb_next = None
        out_idx = 0
        wall0 = time.perf_counter()
        t0_stream = None
        while True:
            dropped = 0
            ok, th, ts = self.thermal.read()
            if not ok:
                break
            if t0_stream is None:
                t0_stream = ts
            if self.realtime:
                # catch up with the wall clock like a live feed would (skip decoded-but-late frames)
                while (ts - t0_stream) / 1000.0 < time.perf_counter() - wall0 - 1.0 / self.fps:
                    ok, ts2 = self.thermal.grab()
                    if not ok:
                        break
                    ts = ts2
                    dropped += 1
                if dropped:
                    ok, th = self.thermal.cap.retrieve()
                    if not ok:
                        break
                ahead = (ts - t0_stream) / 1000.0 - (time.perf_counter() - wall0)
                if ahead > 0:
                    time.sleep(ahead)
            # advance RGB until it passes the master timestamp
            if rgb_next is None:
                ok, f, t = self.rgb.read()
                if not ok:
                    break
                rgb_next = (f, t)
            while rgb_next[1] < ts:
                rgb_prev = rgb_next
                ok, f, t = self.rgb.read()
                if not ok:
                    rgb_next = (rgb_prev[0], float("inf"))
                    break
                rgb_next = (f, t)
            cands = [c for c in (rgb_prev, rgb_next) if c is not None and np.isfinite(c[1])]
            if not cands:
                break
            best = min(cands, key=lambda c: abs(c[1] - ts))
            skew = abs(best[1] - ts)
            if skew > self.tol:
                self.n_skew_violations += 1
            self.n_dropped += dropped
            yield FramePair(out_idx, ts / 1000.0, th, best[0], skew, dropped)
            out_idx += 1
            if self.max_frames and out_idx >= self.max_frames:
                break

    def release(self) -> None:
        self.thermal.release()
        self.rgb.release()
