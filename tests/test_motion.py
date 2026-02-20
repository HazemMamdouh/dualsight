import cv2
import numpy as np

from dualsight.config import load_config
from dualsight.motion.detector import MotionDetector, camera_flow
from dualsight.motion.flow import FlowEstimator
from dualsight.motion.tracker import FusedBlob, MotionTracker, fuse_blobs, iou, project_box


def _bg(h=270, w=480, seed=0):
    rng = np.random.default_rng(seed)
    big = cv2.resize(rng.random((h // 6 + 20, w // 6 + 20)).astype(np.float32), (w + 120, h + 120), interpolation=cv2.INTER_CUBIC)
    return (60 + 120 * (big - big.min()) / (big.max() - big.min())).astype(np.uint8)


def _frame(world, cam, obj_x, h=270, w=480):
    f = world[cam[1]:cam[1] + h, cam[0]:cam[0] + w].copy()
    cv2.rectangle(f, (obj_x, 120), (obj_x + 24, 170), 250, -1)
    return f


def _detector(**over):
    cfg = load_config()
    cfg["motion"].update(over)
    return MotionDetector("t", cfg["motion"], FlowEstimator(cfg["motion"]["flow"], "cpu"))


def test_detects_moving_object_static_camera():
    world = _bg()
    md = _detector()
    res = None
    for k in range(25):
        res = md(_frame(world, (60, 60), 100 + 4 * k))
    assert len(res.blobs) >= 1
    b = max(res.blobs, key=lambda b: b.area)
    cx = (b.box[0] + b.box[2]) / 2
    assert 180 < cx < 215 and b.flow[0] > 2  # moving right


def test_ego_motion_is_compensated():
    world = _bg(seed=2)
    md = _detector()
    res = None
    for k in range(20):
        cam = (60 + 3 * (k % 10), 60)  # camera pans; object is static in the world
        res = md(_frame(world, cam, 200 - 3 * (k % 10)))
    big = [b for b in res.blobs if b.area > 200]
    assert not big, f"static object reported as moving under camera pan: {big}"


def test_camera_flow_translation():
    M = np.float32([[1, 0, 3], [0, 1, -2]])
    f = camera_flow(M, 10, 12)
    assert np.allclose(f[..., 0], 3) and np.allclose(f[..., 1], -2)


def test_fusion_and_tracker():
    a = ([10, 10, 50, 90], [2, 0], 1200.0)
    b = ([14, 12, 52, 95], [2.2, 0], 1100.0)
    c = ([300, 10, 330, 40], [0, 1], 300.0)
    fused = fuse_blobs({"thermal": [a, c], "rgb": [b]})
    assert len(fused) == 2
    both = [f for f in fused if len(f.sources) == 2]
    assert both and both[0].box[0] == 10 and both[0].box[3] == 95

    cfg = load_config()["tracking"]
    tr = MotionTracker(cfg, (640, 360))
    ids = set()
    for k in range(10):
        act = tr.update(k, [FusedBlob([20 + 5 * k, 20, 60 + 5 * k, 100], ["rgb"], [5, 0], 3200)])
        ids |= {t.id for t in act}
    assert len(ids) == 1
    t = next(iter(tr.tracks.values()))
    assert t.confirmed and abs(t.direction_deg() - 0) < 5


def test_geometry_helpers():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    H = np.diag([2.0, 2.0, 1.0])
    assert project_box([1, 2, 3, 4], H) == [2.0, 4.0, 6.0, 8.0]
