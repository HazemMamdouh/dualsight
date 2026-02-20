"""Synthetic, license-clean demo data for DualSight.

Generates a *synced pair* of videos that mimic a perimeter-surveillance rig:

* ``thermal.mp4`` - low-resolution white-hot LWIR channel with drifting fog,
  sensor noise and fixed-pattern noise.
* ``rgb.mp4``     - HD visible channel of the same field of view, affected much
  more strongly by fog and suffering from camera-motion blur while the rig pans
  or shakes.

Moving targets (people, a vehicle, an animal) are real-photo cut-outs taken at
runtime from sample images that ship with ``ultralytics`` and ``scikit-image``
and segmented with a published YOLO11-seg model, so a COCO detector can
genuinely recognise them. No project data is used anywhere.

A ``scenario.json`` ground-truth file (object tracks, fog density, camera
motion) is written next to the videos so the pipeline output can be checked.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _smoothstep(e0: float, e1: float, x: float) -> float:
    t = float(np.clip((x - e0) / max(1e-9, e1 - e0), 0.0, 1.0))
    return t * t * (3 - 2 * t)


def _fractal_noise(h: int, w: int, rng: np.random.Generator, octaves: int = 5,
                   base: int = 4, persistence: float = 0.55) -> np.ndarray:
    """Cheap value-noise in [0, 1]."""
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        cells = base * (2 ** o)
        small = rng.random((max(2, cells * h // w + 1), cells + 1)).astype(np.float32)
        out += amp * cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        total += amp
        amp *= persistence
    out /= total
    out -= out.min()
    out /= max(1e-6, out.max())
    return out


def _ridge(w: int, rng: np.random.Generator, octaves: int = 5) -> np.ndarray:
    line = np.zeros(w, np.float32)
    amp = 1.0
    for o in range(octaves):
        n = 3 * (2 ** o) + 1
        pts = rng.random(n).astype(np.float32)
        line += amp * np.interp(np.linspace(0, n - 1, w), np.arange(n), pts).astype(np.float32)
        amp *= 0.5
    line -= line.min()
    return line / max(1e-6, line.max())


def _tile(tex: np.ndarray, h: int, w: int) -> np.ndarray:
    reps = (int(math.ceil(h / tex.shape[0])), int(math.ceil(w / tex.shape[1])))
    return np.tile(tex, reps + ((1,) if tex.ndim == 3 else ()))[:h, :w]


# --------------------------------------------------------------------------------------
# sprites
# --------------------------------------------------------------------------------------


@dataclass
class Sprite:
    name: str          # person / vehicle / animal
    rgb: np.ndarray    # HxWx3 uint8 (BGR)
    alpha: np.ndarray  # HxW float32 [0, 1]
    heat: np.ndarray   # HxW float32 thermal radiance [0, 255]


def _seg_cutouts(model, path: str, wanted: List[Tuple[str, int]]) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Return (class, bgr_crop, mask_crop) for the requested (class, rank) pairs."""
    img = cv2.imread(path)
    res = model(img, verbose=False, retina_masks=True)[0]
    by_cls: Dict[str, List[Tuple[float, np.ndarray, np.ndarray]]] = {}
    if res.masks is None:
        return []
    for cls, conf, box, m in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist(),
                                 res.boxes.xyxy.cpu().numpy(), res.masks.data.cpu().numpy()):
        name = res.names[int(cls)]
        x1, y1, x2, y2 = [int(v) for v in box]
        H, W = img.shape[:2]
        if x1 <= 3 or y1 <= 3 or x2 >= W - 3 or y2 >= H - 3:
            continue  # truncated by the image border -> not a clean full-body cut-out
        mask = (m > 0.5).astype(np.uint8)
        by_cls.setdefault(name, []).append((conf * (x2 - x1) * (y2 - y1), img[y1:y2, x1:x2].copy(), mask[y1:y2, x1:x2]))
    out = []
    for name, rank in wanted:
        lst = sorted(by_cls.get(name, []), key=lambda t: -t[0])
        if rank < len(lst):
            out.append((name, lst[rank][1], lst[rank][2]))
    return out


def load_sprites(seg_weights: str = "yolo11n-seg.pt") -> Dict[str, List[Sprite]]:
    """Cut real-photo sprites out of package sample images with YOLO11-seg."""
    import skimage.data as skd
    import ultralytics
    from ultralytics import YOLO

    ul_assets = Path(ultralytics.__file__).parent / "assets"
    sk_dir = Path(skd.__file__).parent
    model = YOLO(seg_weights)

    sprites: Dict[str, List[Sprite]] = {"person": [], "vehicle": [], "animal": []}

    for name, bgr, mask in _seg_cutouts(model, str(ul_assets / "bus.jpg"), [("person", 0), ("person", 1)]):
        a = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # skin / exposed areas run hotter than clothes in LWIR
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        skin = ((hsv[..., 0] < 25) & (hsv[..., 1] > 40) & (hsv[..., 2] > 80)).astype(np.float32)
        heat = 192 + 0.12 * g + 38 * cv2.GaussianBlur(skin, (7, 7), 0)
        sprites["person"].append(Sprite("person", bgr, a, np.clip(heat, 0, 255)))

    for name, bgr, mask in _seg_cutouts(model, str(sk_dir / "motorcycle_left.png"), [("motorcycle", 0)]):
        a = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = g.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        engine = np.exp(-(((xx - 0.48 * w) / (0.16 * w)) ** 2 + ((yy - 0.62 * h) / (0.16 * h)) ** 2))
        exhaust = np.exp(-(((xx - 0.72 * w) / (0.12 * w)) ** 2 + ((yy - 0.72 * h) / (0.06 * h)) ** 2))
        heat = 120 + 0.25 * g + 110 * engine + 70 * exhaust
        sprites["vehicle"].append(Sprite("vehicle", bgr, a, np.clip(heat, 0, 255)))

    horse = cv2.imread(str(sk_dir / "horse.png"), cv2.IMREAD_GRAYSCALE)
    if horse is not None:
        mask = (horse < 128).astype(np.float32)
        ys, xs = np.nonzero(mask)
        y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        mask = mask[y1:y2, x1:x2]
        h, w = mask.shape
        rng = np.random.default_rng(7)
        fur = _fractal_noise(h, w, rng, octaves=6, base=8)
        shade = cv2.GaussianBlur(cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5), (0, 0), 6)
        shade = shade / max(1e-6, shade.max())
        base = np.array([30, 62, 112], np.float32)  # bay horse (BGR)
        rgb = base[None, None, :] * (0.55 + 0.35 * fur[..., None] + 0.5 * shade[..., None])
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        heat = 196 + 30 * shade + 10 * fur
        sprites["animal"].append(Sprite("animal", rgb, cv2.GaussianBlur(mask, (5, 5), 0), np.clip(heat, 0, 255)))
    return sprites


# --------------------------------------------------------------------------------------
# scenario
# --------------------------------------------------------------------------------------


@dataclass
class Actor:
    kind: str
    sprite_idx: int
    t0: float
    t1: float
    x0: float          # normalised world-x of the foot point at t0
    x1: float
    y_foot: float      # normalised world-y of the foot point
    height: float      # normalised sprite height in the world image
    flip: bool = False
    bob: float = 0.0   # vertical bob amplitude (normalised)
    gait_hz: float = 1.8

    def pos(self, t: float) -> Optional[Tuple[float, float]]:
        if not (self.t0 <= t <= self.t1):
            return None
        a = (t - self.t0) / (self.t1 - self.t0)
        x = self.x0 + (self.x1 - self.x0) * a
        y = self.y_foot - abs(math.sin(math.pi * self.gait_hz * (t - self.t0))) * self.bob
        return x, y


@dataclass
class Scenario:
    fps: float = 15.0
    duration: float = 48.0
    rgb_size: Tuple[int, int] = (1280, 720)
    thermal_size: Tuple[int, int] = (640, 360)
    margin: Tuple[int, int] = (140, 70)
    seed: int = 2024
    actors: List[Actor] = field(default_factory=list)

    # fog: (t_on_start, t_on_end, t_off_start, t_off_end, peak density)
    fog: Tuple[float, float, float, float, float] = (10.0, 14.0, 29.0, 34.0, 1.0)
    # camera motion segments: (t0, t1, kind)
    camera: List[Tuple[float, float, str]] = field(default_factory=list)

    @staticmethod
    def default() -> "Scenario":
        s = Scenario()
        s.actors = [
            Actor("person", 0, 3.0, 15.0, -0.04, 1.04, 0.61, 0.17, flip=False, bob=0.004),
            Actor("vehicle", 0, 16.0, 25.0, 1.10, -0.12, 0.775, 0.17, flip=True, bob=0.001, gait_hz=4.0),
            Actor("animal", 0, 26.5, 36.5, -0.10, 1.08, 0.68, 0.19, flip=False, bob=0.006, gait_hz=2.4),
            Actor("person", 1, 36.0, 46.5, 1.06, -0.06, 0.86, 0.26, flip=True, bob=0.005),
            Actor("person", 0, 36.6, 47.0, 1.13, 0.02, 0.88, 0.27, flip=True, bob=0.005),
        ]
        s.camera = [(6.0, 10.0, "pan"), (20.0, 23.0, "pan"), (38.5, 42.5, "shake")]
        return s

    def fog_density(self, t: float) -> float:
        a, b, c, d, peak = self.fog
        return peak * (_smoothstep(a, b, t) - _smoothstep(c, d, t))

    def camera_offset(self, t: float, rng_tab: np.ndarray) -> Tuple[float, float]:
        mx, my = self.margin
        x, y = float(mx), float(my)
        for t0, t1, kind in self.camera:
            if not (t0 <= t <= t1):
                continue
            u = (t - t0) / (t1 - t0)
            env = math.sin(math.pi * u)
            if kind == "pan":
                x += 0.9 * mx * math.sin(2 * math.pi * u) * env
                y += 0.15 * my * math.sin(4 * math.pi * u)
            else:  # hand-held shake: band-limited random walk, deterministic from the table
                k = t * 4.0  # ~4 Hz hand-held / vehicle-mount shake
                i0 = int(k) % (len(rng_tab) - 1)
                f = k - int(k)
                jx = (1 - f) * rng_tab[i0, 0] + f * rng_tab[i0 + 1, 0]
                jy = (1 - f) * rng_tab[i0, 1] + f * rng_tab[i0 + 1, 1]
                x += 0.30 * mx * jx * env
                y += 0.30 * my * jy * env
        return float(x), float(y)


# --------------------------------------------------------------------------------------
# renderer
# --------------------------------------------------------------------------------------


class SceneRenderer:
    def __init__(self, sc: Scenario, sprites: Dict[str, List[Sprite]]):
        self.sc = sc
        self.sprites = sprites
        self.rng = np.random.default_rng(sc.seed)
        W, H = sc.rgb_size
        mx, my = sc.margin
        self.WW, self.WH = W + 2 * mx, H + 2 * my
        self.shake_tab = self.rng.uniform(-1, 1, (4096, 2)).astype(np.float32)
        self._build_world()
        tw, th = sc.thermal_size
        self.fpn = (self.rng.normal(0, 1.0, (1, tw)) + self.rng.normal(0, 0.5, (th, tw))).astype(np.float32)

    # ---- static world --------------------------------------------------------------
    def _build_world(self) -> None:
        import skimage.data as skd

        WW, WH = self.WW, self.WH
        rng = self.rng
        yy, xx = np.mgrid[0:WH, 0:WW].astype(np.float32)
        hor = 0.36 * WH
        ridge_far = hor - 0.10 * WH * _ridge(WW, rng)
        ridge_near = hor + 0.03 * WH - 0.06 * WH * _ridge(WW, rng)
        self.hor = hor

        grass = skd.grass().astype(np.float32) / 255.0
        gravel = skd.gravel().astype(np.float32) / 255.0
        grass_t = _tile(cv2.resize(grass, (256, 256)), WH, WW)
        gravel_t = _tile(cv2.resize(gravel, (200, 200)), WH, WW)
        n_lo = _fractal_noise(WH, WW, rng, octaves=4, base=3)

        sky = yy < ridge_far[None, :]
        far = (~sky) & (yy < ridge_near[None, :])
        ground = ~(sky | far)
        road = (yy > 0.735 * WH) & (yy < 0.815 * WH)

        # ---------------- RGB
        rgb = np.zeros((WH, WW, 3), np.float32)
        g = np.clip(yy / hor, 0, 1)[..., None]
        rgb[:] = (1 - g) * np.array([200, 150, 95]) + g * np.array([235, 215, 190])  # BGR sky
        far_col = np.array([120, 125, 105]) * (0.85 + 0.25 * n_lo[..., None])
        rgb = np.where(far[..., None], far_col, rgb)
        gcol = np.array([55, 105, 95]) * (0.55 + 0.55 * grass_t[..., None]) * (0.8 + 0.4 * n_lo[..., None])
        dirt = np.array([95, 135, 160]) * (0.7 + 0.4 * gravel_t[..., None])
        mixmap = np.clip((n_lo - 0.45) * 4, 0, 1)[..., None]
        gcol = (1 - mixmap) * gcol + mixmap * dirt
        rgb = np.where(ground[..., None], gcol, rgb)
        roadcol = np.array([120, 140, 150]) * (0.75 + 0.35 * gravel_t[..., None])
        rgb = np.where(road[..., None], roadcol, rgb)

        # ---------------- thermal (radiance, white-hot)
        th = np.zeros((WH, WW), np.float32)
        th[:] = 38 + 18 * (yy / hor)
        th = np.where(far, 92 + 10 * n_lo, th)
        th = np.where(ground, 102 + 20 * n_lo + 10 * grass_t, th)
        th = np.where(road, 140 + 8 * gravel_t, th)

        # bushes / rocks
        for _ in range(26):
            cx = rng.uniform(0, WW)
            cy = rng.uniform(hor + 0.06 * WH, WH * 0.98)
            r = rng.uniform(10, 34) * (0.5 + (cy - hor) / (WH - hor))
            is_rock = rng.random() < 0.35
            m = (((xx - cx) / r) ** 2 + ((yy - cy) / (0.7 * r)) ** 2) < 1.0
            if road[int(min(cy, WH - 1)), int(min(cx, WW - 1))]:
                continue
            if is_rock:
                rgb[m] = np.array([140, 145, 150]) * (0.6 + 0.5 * gravel_t[m][:, None])
                th[m] = 150 + 15 * gravel_t[m]
            else:
                rgb[m] = np.array([40, 85, 60]) * (0.6 + 0.6 * grass_t[m][:, None])
                th[m] = 96 + 10 * grass_t[m]

        # border fence: posts + three wires
        fy = 0.545 * WH
        for px in np.arange(10, WW, 58):
            ph = 0.075 * WH
            cv2.rectangle(rgb, (int(px), int(fy - ph)), (int(px + 4), int(fy)), (60, 70, 80), -1)
            cv2.rectangle(th, (int(px), int(fy - ph)), (int(px + 4), int(fy)), 128, -1)
        for k in range(3):
            y = int(fy - 0.02 * WH - k * 0.02 * WH)
            cv2.line(rgb, (0, y), (WW, y), (90, 95, 100), 1, cv2.LINE_AA)
            cv2.line(th, (0, y), (WW, y), 118, 1, cv2.LINE_AA)

        self.world_rgb = np.clip(rgb, 0, 255).astype(np.float32)
        self.world_th = np.clip(th, 0, 255).astype(np.float32)
        self.fog_field = _fractal_noise(WH, WW * 2, rng, octaves=4, base=3)  # drifts horizontally

    # ---- per-frame composition ------------------------------------------------------
    def _paste(self, dst_rgb: np.ndarray, dst_th: np.ndarray, sp: Sprite, cx: float, foot_y: float,
               height_px: float, flip: bool) -> Tuple[int, int, int, int]:
        h0, w0 = sp.alpha.shape
        s = height_px / h0
        w, h = max(2, int(w0 * s)), max(2, int(h0 * s))
        rgb = cv2.resize(sp.rgb, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
        a = cv2.resize(sp.alpha, (w, h), interpolation=cv2.INTER_AREA)
        heat = cv2.resize(sp.heat, (w, h), interpolation=cv2.INTER_AREA)
        if flip:
            rgb, a, heat = rgb[:, ::-1], a[:, ::-1], heat[:, ::-1]
        x1, y1 = int(cx - w / 2), int(foot_y - h)
        x2, y2 = x1 + w, y1 + h
        H, W = dst_th.shape
        sx1, sy1 = max(0, -x1), max(0, -y1)
        dx1, dy1, dx2, dy2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        if dx2 <= dx1 or dy2 <= dy1:
            return (0, 0, 0, 0)
        ww, hh = dx2 - dx1, dy2 - dy1
        aa = a[sy1:sy1 + hh, sx1:sx1 + ww]
        dst_rgb[dy1:dy2, dx1:dx2] = dst_rgb[dy1:dy2, dx1:dx2] * (1 - aa[..., None]) + rgb[sy1:sy1 + hh, sx1:sx1 + ww] * aa[..., None]
        # thermal: slight heat halo
        dst_th[dy1:dy2, dx1:dx2] = dst_th[dy1:dy2, dx1:dx2] * (1 - aa) + heat[sy1:sy1 + hh, sx1:sx1 + ww] * aa
        return dx1, dy1, dx2, dy2

    def render(self, t: float) -> Tuple[np.ndarray, np.ndarray, Dict]:
        sc = self.sc
        W, H = sc.rgb_size
        WW, WH = self.WW, self.WH
        rgb = self.world_rgb.copy()
        th = self.world_th.copy()

        gt_objs = []
        for i, act in enumerate(sc.actors):
            p = act.pos(t)
            if p is None:
                continue
            sp = self.sprites[act.kind][act.sprite_idx % len(self.sprites[act.kind])]
            box = self._paste(rgb, th, sp, p[0] * WW, p[1] * WH, act.height * WH, act.flip)
            gt_objs.append({"id": i, "kind": act.kind, "world_box": list(map(int, box))})

        # fog (applied in world space so it moves with the scene)
        dens = sc.fog_density(t)
        if dens > 1e-3:
            shift = int((t * 22.0) % WW)
            ff = self.fog_field[:, shift:shift + WW]
            yy = np.linspace(0, 1, WH, dtype=np.float32)[:, None]
            depth = np.clip(1.25 - yy, 0.25, 1.0)  # far = denser
            d = dens * depth * (0.55 + 0.9 * ff)
            t_rgb = np.exp(-3.4 * d)[..., None]
            t_th = np.exp(-1.15 * d)
            rgb = rgb * t_rgb + np.array([205, 205, 200], np.float32) * (1 - t_rgb)
            th = th * t_th + 118.0 * (1 - t_th)

        # camera exposure: average sub-frame crops along the camera path
        cams = []
        n_sub = 9
        exp = 0.85 / sc.fps
        for k in range(n_sub):
            tt = t - exp / 2 + exp * k / (n_sub - 1)
            cams.append(sc.camera_offset(tt, self.shake_tab))
        cx, cy = sc.camera_offset(t, self.shake_tab)

        def crop(img, ox, oy):
            M = np.float32([[1, 0, -ox], [0, 1, -oy]])
            return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        moving = max(abs(a[0] - cams[0][0]) + abs(a[1] - cams[0][1]) for a in cams) > 0.5
        if moving:
            out_rgb = np.mean([crop(rgb, ox, oy) for ox, oy in cams], axis=0)
            # thermal microbolometers integrate over a shorter effective window here
            th_c = cams[n_sub // 2 - 1: n_sub // 2 + 2]
            out_th = np.mean([crop(th, ox, oy) for ox, oy in th_c], axis=0)
        else:
            out_rgb = crop(rgb, cx, cy)
            out_th = crop(th, cx, cy)

        # sensor models
        out_rgb = out_rgb + self.rng.normal(0, 2.0, out_rgb.shape)
        out_rgb = np.clip(out_rgb, 0, 255).astype(np.uint8)
        tw, thh = sc.thermal_size
        out_th = cv2.resize(out_th, (tw, thh), interpolation=cv2.INTER_AREA)
        out_th = cv2.GaussianBlur(out_th, (0, 0), 0.7)
        out_th = out_th + self.fpn + self.rng.normal(0, 2.0, out_th.shape)
        out_th = np.clip(out_th, 0, 255).astype(np.uint8)

        for o in gt_objs:
            x1, y1, x2, y2 = o["world_box"]
            o["rgb_box"] = [int(x1 - cx), int(y1 - cy), int(x2 - cx), int(y2 - cy)]
        meta = {"t": round(t, 4), "fog": round(dens, 4), "camera": [round(cx, 2), round(cy, 2)],
                "camera_moving": bool(moving), "objects": gt_objs}
        return out_th, out_rgb, meta


def _open_writer(path: str, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    for cc in ("avc1", "mp4v"):
        wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*cc), fps, size)
        if wr.isOpened():
            return wr
    raise RuntimeError(f"cannot open writer for {path}")


def generate_demo(out_dir: str, seconds: Optional[float] = None, fps: Optional[float] = None,
                  seg_weights: str = "yolo11n-seg.pt", progress: Optional[Callable[[int, int], None]] = None) -> Dict[str, str]:
    """Render the default scenario to ``out_dir`` and return the written paths."""
    os.makedirs(out_dir, exist_ok=True)
    sc = Scenario.default()
    if fps:
        sc.fps = fps
    if seconds:
        sc.duration = seconds
    sprites = load_sprites(seg_weights)
    rend = SceneRenderer(sc, sprites)
    n = int(round(sc.duration * sc.fps))
    p_th, p_rgb = os.path.join(out_dir, "thermal.mp4"), os.path.join(out_dir, "rgb.mp4")
    from ..io.writers import VideoWriter

    w_th = VideoWriter(p_th, sc.fps, crf=16)
    w_rgb = VideoWriter(p_rgb, sc.fps, crf=18)
    gt = []
    for i in range(n):
        t = i / sc.fps
        th, rgb, meta = rend.render(t)
        w_th.write(cv2.cvtColor(th, cv2.COLOR_GRAY2BGR))
        w_rgb.write(rgb)
        meta["frame"] = i
        gt.append(meta)
        if progress:
            progress(i + 1, n)
    w_th.release()
    w_rgb.release()
    p_gt = os.path.join(out_dir, "scenario.json")
    with open(p_gt, "w") as f:
        json.dump({"fps": sc.fps, "duration": sc.duration, "rgb_size": sc.rgb_size,
                   "thermal_size": sc.thermal_size, "fog": sc.fog, "camera": sc.camera,
                   "actors": [a.__dict__ for a in sc.actors], "frames": gt}, f, default=float)
    return {"thermal": p_th, "rgb": p_rgb, "scenario": p_gt}
