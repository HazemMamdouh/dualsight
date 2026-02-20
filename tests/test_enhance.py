import cv2
import numpy as np

from dualsight.config import DEFAULTS, load_config
from dualsight.enhance.deblur import MotionDeblur, motion_psf
from dualsight.enhance.quality import CameraMotionGate, FogGate, HysteresisGate, fog_metrics
from dualsight.enhance.thermal import ThermalEnhancer, destripe


def _scene(h=360, w=640, seed=0):
    rng = np.random.default_rng(seed)
    img = cv2.resize(rng.random((h // 8, w // 8)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    img = (img - img.min()) / (img.max() - img.min())
    img = (40 + 180 * img).astype(np.float32)
    for _ in range(40):
        x, y = rng.integers(0, w), rng.integers(0, h)
        cv2.circle(img, (int(x), int(y)), int(rng.integers(4, 18)), float(rng.integers(0, 255)), -1)
    return img


def _fog(img, t=0.3, A=125.0):
    return np.clip(img * t + A * (1 - t), 0, 255).astype(np.uint8)


def test_fog_score_increases_with_veil():
    img = _scene()
    clear = fog_metrics(img.astype(np.uint8))["fog_score"]
    foggy = fog_metrics(_fog(img, 0.25))["fog_score"]
    assert foggy > clear + 0.3


def test_hysteresis_gate_holds():
    g = HysteresisGate(0.5, 0.4, hold=3)
    seq = [0.6, 0.6, 0.6, 0.45, 0.45, 0.3, 0.3, 0.3]
    out = [g.update(v) for v in seq]
    assert out[:2] == [False, False] and out[2] is True
    assert out[3] is True and out[4] is True  # inside the band -> keeps state
    assert out[-1] is False


def test_fog_gate_toggles():
    gate = FogGate(DEFAULTS["thermal"]["gate"])
    img = _scene()
    states = [gate(_fog(img, 0.1))[0] for _ in range(15)]
    assert states[-1] is True
    states = [gate(img.astype(np.uint8))[0] for _ in range(20)]
    assert states[-1] is False


def test_classical_enhancer_restores_contrast():
    cfg = load_config()["thermal"]
    cfg["enhancer"] = "classical"
    enh = ThermalEnhancer(cfg, "cpu")
    foggy = _fog(_scene(), 0.25)
    out = enh(foggy, 0.6)
    assert out.shape == foggy.shape
    assert fog_metrics(out)["spread"] > 2 * fog_metrics(foggy)["spread"]


def test_destripe_removes_column_noise():
    rng = np.random.default_rng(1)
    base = np.full((200, 300), 120, np.float32)
    stripes = base + rng.normal(0, 3, (1, 300))
    out = destripe(np.clip(stripes, 0, 255).astype(np.uint8))
    assert out.astype(np.float32).mean(axis=0).std() < 0.5 * stripes.mean(axis=0).std()


def test_wiener_deblur_improves_sharpness():
    img = cv2.cvtColor(_scene().astype(np.uint8), cv2.COLOR_GRAY2BGR)
    dx, dy = 14.0, 3.0
    cfg = dict(DEFAULTS["rgb"]["deblur"])
    deb = MotionDeblur(cfg, "cpu")
    psf, L, _ = deb.psf_for_shift(dx, dy)
    blurred = cv2.filter2D(img, -1, psf, borderType=cv2.BORDER_REFLECT)
    restored = deb(blurred, dx, dy)
    err_b = np.abs(blurred.astype(float) - img)[40:-40, 40:-40].mean()
    err_r = np.abs(restored.astype(float) - img)[40:-40, 40:-40].mean()
    assert err_r < 0.8 * err_b


def test_psf_normalised():
    k = motion_psf(9.5, 0.4, 13)
    assert abs(k.sum() - 1) < 1e-5 and k.max() < 1


def test_camera_gate_measures_global_shift():
    img = _scene(seed=3).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cfg = dict(DEFAULTS["rgb"]["gate"])
    cfg["hold_frames"] = 1
    gate = CameraMotionGate(cfg)
    gate(img)
    M = np.float32([[1, 0, 9], [0, 1, -4]])
    moved = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)
    on, cm = gate(moved)
    assert on
    assert abs(cm.dx - 9) < 1.0 and abs(cm.dy + 4) < 1.0


def test_camera_gate_static_with_moving_object():
    img = cv2.cvtColor(_scene(seed=4).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    gate = CameraMotionGate(DEFAULTS["rgb"]["gate"])
    gate(img)
    nxt = img.copy()
    nxt[100:220, 300:380] = np.roll(img[100:220, 300:380], 12, axis=1)  # a target moves, camera does not
    on, cm = gate(nxt)
    assert not on and cm.magnitude < 1.0
