"""Configuration: a nested dict of defaults, overridable from YAML and the CLI."""
from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import yaml

DEFAULTS: Dict[str, Any] = {
    "device": "auto",                 # auto | cpu | cuda | cuda:N
    "sources": {
        "use_gstreamer": "auto",      # auto | true | false
        "hw_decode": "none",          # none | nvidia (desktop NVDEC) | jetson (nvv4l2decoder)
        "sync_tolerance_ms": 40.0,    # max |t_thermal - t_rgb| to accept a pair
        "realtime": False,            # pace offline files like a live feed (drop frames when behind)
        "max_frames": 0,              # 0 = all
        "start_s": 0.0,
    },
    "registration": {
        # 3x3 homography mapping thermal pixels -> RGB pixels. null = pure scale (co-boresighted rig)
        "thermal_to_rgb": None,
    },
    "thermal": {
        "gate": {                      # classical fog / low-contrast check
            "on_score": 0.40,          # fog score above which enhancement is switched on
            "off_score": 0.30,         # ... and below which it is switched off (hysteresis)
            "hold_frames": 5,          # consecutive frames needed to flip state
            "spread_ref": 0.55,        # p1..p99 intensity spread of a clear scene (fraction of 255)
            "grad_ref": 9.0,           # mean Sobel magnitude of a clear scene
            "force": None,             # None | true | false  - bypass the gate
        },
        "enhancer": "auto",            # auto | classical | esrgan | hybrid | off
        "dehaze": {
            "omega": 0.85,             # haze removal strength at full fog score
            "t_min": 0.4,
            "window": 31,              # min-filter window (thermal px)
            "guided_radius": 24,
            "guided_eps": 1e-3,
            "clahe_clip": 1.4,
            "clahe_grid": 8,
            "unsharp": 0.4,
            "denoise": 2,              # bilateral d (0 = off)
            "temporal_nr": 0.35,       # motion-adaptive recursive NR, min blend weight (0 = off)
            "destripe": False,         # column-FPN removal, only for cores without a working NUC
        },
        "esrgan": {
            "model": "realesr-general-x4v3",   # or RealESRGAN_x4plus (heavier, sharper)
            "weights_dir": "~/.cache/dualsight",
            "tile": 0,                 # 0 = whole frame; >0 for low-VRAM devices (e.g. 256 on Jetson)
            "half": True,
            "keep_sr_resolution": False,  # False: x4 output is area-downsampled back (detail-preserving denoise)
        },
    },
    "rgb": {
        "gate": {                      # classical camera-motion check
            "shift_px": 2.5,           # global inter-frame shift (full-res px) that counts as camera motion
            "hold_frames": 2,
            "require_blur": False,     # additionally require a sharpness drop vs the running baseline
            "blur_ratio": 0.75,
            "analysis_width": 480,
            "force": None,
        },
        "deblur": {
            "method": "wiener",        # wiener | off
            "exposure": 0.85,          # fraction of the frame interval the shutter is open
            "max_len": 61,             # max PSF length (px)
            "nsr": 0.02,               # noise-to-signal ratio for Wiener
            "dering": 5,               # clamp to the local min/max of the input (window px, 0 = off)
            "taper": 24,               # edge taper (px) against FFT ringing
        },
    },
    "motion": {
        "proc_width": 480,             # both channels are analysed at this width
        "flow": {
            "backend": "auto",         # auto | dis | farneback | cuda_farneback | nvof | raft
            "dis_preset": "medium",
            "raft_iters": 12,
        },
        "ego": {
            "enabled": True,
            "share_rig_motion": True,  # use the RGB corner-based camera estimate for both channels
            "grid_step": 12,
            "ransac_px": 1.5,
            "resid_k": 0.25,           # extra residual threshold per px/frame of camera motion
        },
        "flow_thresh": 0.4,            # residual flow magnitude (proc px / frame) considered motion
        "flow_k_mad": 4.0,             # adaptive part: median + k * MAD
        "bg": {
            "method": "mog2",          # mog2 | knn | diff
            "history": 200,
            "var_threshold": 24,
            "diff_thresh": 18,         # aligned frame differencing (used while the camera moves)
            "warmup_frames": 10,
        },
        "min_blob_snr": 1.3,           # blob mean residual flow must be >= this x the threshold
        "transition_guard": {          # fog rolling in / clearing: global radiance drift breaks brightness constancy
            "fog_rate": 0.08,          # |d fog_score / dt| (per second) that marks a transition
            "snr_boost": 2.0,          # min_blob_snr multiplier while in transition
        },
        "combine": "and",              # and | or  - how flow & background masks are fused per channel
        "min_area_frac": 0.00035,      # min blob area as fraction of the frame
        "max_area_frac": 0.12,         # larger blobs are scene-wide changes (fog fronts, gain jumps)
        "max_width_frac": 0.6,         # ... as are blobs spanning most of the frame width
        "morph": 5,
        "channels": ["thermal", "rgb"],
    },
    "tracking": {
        "iou_match": 0.15,
        "max_center_dist": 0.05,       # fraction of frame diagonal (per frame, after prediction)
        "max_area_ratio": 3.0,         # reject matches whose box area changes more than this
        "max_age": 12,                 # frames a track survives without a blob
        "min_hits": 4,                 # frames before a track becomes an event
        "min_event_s": 1.0,
        "min_displacement_frac": 0.03, # unlabeled tracks must travel >= this fraction of the diagonal
        "merge_dist": 0.05,            # fragment consolidation: max gap (fraction of diagonal)
    },
    "detector": {
        "enabled": True,
        "model": "yolo11n.pt",         # any Ultralytics detection model (COCO classes are mapped below)
        "conf": 0.25,
        "imgsz": 640,
        "every": 3,                    # run every N frames while motion is active
        "mode": "roi",                 # roi | full
        "roi_pad": 1.6,
        "roi_min": 192,
        "max_rois": 6,
        "on_channels": ["rgb", "thermal"],
        "half": True,
    },
    "analytics": {
        "heatmap_width": 320,
        "grid": [4, 3],                # zone grid (cols, rows) for the 'where' table
    },
    "viz": {
        "panel_width": 640,
        "timeline_s": 20.0,
        "thermal_cmap": "gray",        # gray | inferno | turbo
        "show_masks": True,
    },
}


def deep_update(base: Dict[str, Any], upd: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not upd:
        return base
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path) as f:
            deep_update(cfg, yaml.safe_load(f) or {})
    deep_update(cfg, overrides or {})
    return cfg


def set_dotted(cfg: Dict[str, Any], dotted: str, value: str) -> None:
    """Apply ``a.b.c=value`` style overrides (value parsed as YAML)."""
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = yaml.safe_load(value)
