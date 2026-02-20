import os

import cv2
import numpy as np

from dualsight.cli import build_parser
from dualsight.config import load_config, set_dotted
from dualsight.io.sources import SyncedDualSource, build_gst_pipeline


def _write(path, fps, n, size):
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(n):
        f = np.zeros((size[1], size[0], 3), np.uint8)
        cv2.putText(f, str(i), (10, size[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        w.write(f)
    w.release()


def test_sync_pairs_different_rates(tmp_path):
    th, rgb = str(tmp_path / "t.mp4"), str(tmp_path / "r.mp4")
    _write(th, 10, 20, (160, 120))
    _write(rgb, 20, 40, (320, 240))
    cfg = load_config(overrides={"sources": {"use_gstreamer": False}})
    src = SyncedDualSource(th, rgb, cfg)
    pairs = list(src)
    src.release()
    assert len(pairs) == 20
    assert max(p.skew_ms for p in pairs) <= 30
    assert pairs[0].thermal.shape[:2] == (120, 160) and pairs[0].rgb.shape[:2] == (240, 320)


def test_gst_pipeline_strings():
    assert "filesrc" in build_gst_pipeline("/data/a.mp4")
    assert "nvv4l2decoder" in build_gst_pipeline("/data/a.mp4", "jetson")
    assert build_gst_pipeline("rtsp://cam/1").startswith("rtspsrc")
    assert "udpsrc port=5000" in build_gst_pipeline("udp://0.0.0.0:5000")
    assert build_gst_pipeline("gst:videotestsrc ! appsink") == "videotestsrc ! appsink"


def test_config_overrides():
    cfg = load_config()
    set_dotted(cfg, "motion.combine", "or")
    set_dotted(cfg, "detector.every", "5")
    assert cfg["motion"]["combine"] == "or" and cfg["detector"]["every"] == 5


def test_cli_parses():
    p = build_parser()
    a = p.parse_args(["run", "--thermal", "t.mp4", "--rgb", "r.mp4", "--device", "cuda", "--show", "--save-video"])
    assert a.cmd == "run" and a.show and a.save_video and a.device == "cuda"
    a = p.parse_args(["render", "--run", "runs/x", "--compare"])
    assert a.compare and a.view == "enhanced"


def test_pair_folders(tmp_path):
    from dualsight.synth.real_data import pair_folders

    for d in ("lwir", "visible"):
        (tmp_path / d).mkdir()
        for i in range(6):
            cv2.imwrite(str(tmp_path / d / f"I{i:05d}.jpg"), np.full((64, 80, 3), 40 * i % 255, np.uint8))
    out = pair_folders(str(tmp_path / "lwir"), str(tmp_path / "visible"), str(tmp_path / "out"), 20.0)
    cap = cv2.VideoCapture(out["thermal"])
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 6
