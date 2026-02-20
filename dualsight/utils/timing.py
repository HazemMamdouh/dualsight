"""Tiny per-stage profiler."""
from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List

import numpy as np


class StageTimer:
    def __init__(self) -> None:
        self.samples: Dict[str, List[float]] = defaultdict(list)
        self.current: Dict[str, float] = {}

    def start_frame(self) -> None:
        self.current = {}

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = (time.perf_counter() - t0) * 1000.0
            self.current[name] = self.current.get(name, 0.0) + dt
            self.samples[name].append(dt)

    def summary(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for k, v in self.samples.items():
            a = np.asarray(v)
            out[k] = {"mean_ms": float(a.mean()), "p50_ms": float(np.median(a)),
                      "p95_ms": float(np.percentile(a, 95)), "calls": int(a.size)}
        return out
