"""Dense optical flow with interchangeable CPU / GPU back-ends.

=================  ======  ====================================================
backend            device  notes
=================  ======  ====================================================
``dis``            CPU     OpenCV DIS - fast, robust default on CPU / Jetson CPU
``farneback``      CPU     classic polynomial-expansion flow
``cuda_farneback`` GPU     OpenCV CUDA build required (``cv2.cuda``)
``nvof``           GPU     NVIDIA Optical Flow SDK (Turing+), OpenCV CUDA build
``raft``           GPU     torchvision RAFT-small (published weights), any CUDA torch
=================  ======  ====================================================

``auto`` picks ``cuda_farneback`` > ``raft`` > ``dis`` depending on what the
machine offers and on ``--device``.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from ..utils.device import is_gpu, opencv_cuda_available

log = logging.getLogger("dualsight")


class FlowEstimator:
    def __init__(self, cfg: dict, device: str) -> None:
        backend = cfg.get("backend", "auto")
        if backend == "auto":
            if is_gpu(device) and device.startswith("cuda") and opencv_cuda_available():
                backend = "cuda_farneback"
            elif is_gpu(device):
                backend = "raft"
            else:
                backend = "dis"
        if backend in ("cuda_farneback", "nvof") and not opencv_cuda_available():
            log.warning("%s needs an OpenCV CUDA build -> using %s", backend, "raft" if is_gpu(device) else "dis")
            backend = "raft" if is_gpu(device) else "dis"
        self.backend = backend
        self.device = device
        self.cfg = cfg
        self._nvof = None
        if backend == "raft":
            try:
                import torch
                from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

                self.torch = torch
                self.raft = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).to(device).eval()
                self.iters = int(cfg.get("raft_iters", 12))
            except Exception as e:  # weights not downloadable, old torchvision, ...
                log.warning("RAFT unavailable (%s) -> falling back to DIS", str(e).splitlines()[0][:120])
                backend = self.backend = "dis"
        if backend == "dis":
            preset = {"ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST, "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
                      "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM}[cfg.get("dis_preset", "medium")]
            self.dis = cv2.DISOpticalFlow_create(preset)
        elif backend == "cuda_farneback":
            self.cfb = cv2.cuda.FarnebackOpticalFlow_create(5, 0.5, False, 15, 3, 5, 1.2, 0)
        log.info("optical flow backend: %s", self.backend)

    def __call__(self, prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
        """Flow (H x W x 2, float32, pixels/frame) from ``prev`` to ``cur`` (uint8 gray)."""
        if self.backend == "dis":
            return self.dis.calc(prev, cur, None)
        if self.backend == "farneback":
            return cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 4, 15, 3, 5, 1.2, 0)
        if self.backend == "cuda_farneback":
            g0, g1 = cv2.cuda_GpuMat(), cv2.cuda_GpuMat()
            g0.upload(prev)
            g1.upload(cur)
            return self.cfb.calc(g0, g1, None).download()
        if self.backend == "nvof":
            if self._nvof is None:
                h, w = prev.shape
                self._nvof = cv2.cuda.NvidiaOpticalFlow_2_0.create((w, h), 5, 1, 1, False, False, False, 0)
            flow, _ = self._nvof.calc(prev, cur, None)
            return self._nvof.convertToFloat(flow, None)
        return self._raft(prev, cur)

    def _raft(self, prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
        torch = self.torch
        h, w = prev.shape
        H8, W8 = (h + 7) // 8 * 8, (w + 7) // 8 * 8

        def prep(g):
            x = torch.from_numpy(cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)).permute(2, 0, 1)[None].float()
            x = torch.nn.functional.interpolate(x, size=(H8, W8), mode="bilinear", align_corners=False)
            return (x / 127.5 - 1.0).to(self.device)

        with torch.no_grad():
            flows = self.raft(prep(prev), prep(cur), num_flow_updates=self.iters)
        f = flows[-1][0].permute(1, 2, 0).cpu().numpy()
        f = cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR)
        f[..., 0] *= w / W8
        f[..., 1] *= h / H8
        return f.astype(np.float32)
