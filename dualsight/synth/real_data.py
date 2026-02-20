"""Real, synchronised thermal + visible footage for demos and benchmarks.

``dualsight fetch-data vot-rgbt --seq fog6,walkingnight`` downloads aligned colour + thermal
sequences of the VOT-RGBT challenge (a subset of RGBT234): *static surveillance cameras* with real
fog, night and rain - the closest public match to a perimeter-monitoring head.

``dualsight fetch-data kaist`` downloads one sequence of the **KAIST Multispectral Pedestrian
Benchmark** (Hwang et al., CVPR 2015) - a co-boresighted 640x512 LWIR + visible rig recording real
streets at 20 fps, day (set00-02) and night (set03-05) - from a public Hugging Face mirror, and
turns its frame folders into the two synced input streams DualSight expects.

KAIST is released for research use (CC BY-NC-SA 4.0 on the original project page); keep the
attribution when you publish results, and do not commit the footage to the repository.

Any other paired dataset whose frames are stored as two folders of index-aligned images can be
converted with ``dualsight frames-to-video``.
"""
from __future__ import annotations

import glob
import logging
import os
from typing import Dict, List, Optional

import cv2

from ..io.writers import VideoWriter

log = logging.getLogger("dualsight")

KAIST_REPO = "richidubey/KAIST-Multispectral-Pedestrian-Detection-Dataset"
KAIST_SETS = {
    "set00": "day · campus", "set01": "day · road", "set02": "day · downtown",
    "set03": "night · campus", "set04": "night · road", "set05": "night · downtown",
}
KAIST_FPS = 20.0


def frames_to_video(frames: List[str], out: str, fps: float, gray: bool = False, max_frames: int = 0) -> int:
    w = VideoWriter(out, fps, crf=17)
    n = 0
    for p in frames[: max_frames or None]:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
        if img is None:
            continue
        w.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if gray else img)
        n += 1
    w.release()
    return n


def pair_folders(thermal_dir: str, rgb_dir: str, out_dir: str, fps: float, max_frames: int = 0,
                 pattern: str = "*") -> Dict[str, str]:
    """Convert two index-aligned frame folders into ``thermal.mp4`` + ``rgb.mp4``."""
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    th = sorted(p for p in glob.glob(os.path.join(thermal_dir, pattern)) if p.lower().endswith(exts))
    vi = sorted(p for p in glob.glob(os.path.join(rgb_dir, pattern)) if p.lower().endswith(exts))
    names_t = {os.path.basename(p) for p in th}
    common = [os.path.basename(p) for p in vi if os.path.basename(p) in names_t]
    if common and len(common) >= 0.9 * min(len(th), len(vi)):
        th = [os.path.join(thermal_dir, c) for c in common]
        vi = [os.path.join(rgb_dir, c) for c in common]
    else:
        n = min(len(th), len(vi))
        th, vi = th[:n], vi[:n]
    if not th:
        raise FileNotFoundError(f"no paired frames in {thermal_dir} / {rgb_dir}")
    os.makedirs(out_dir, exist_ok=True)
    pt, pr = os.path.join(out_dir, "thermal.mp4"), os.path.join(out_dir, "rgb.mp4")
    nt = frames_to_video(th, pt, fps, gray=True, max_frames=max_frames)
    nr = frames_to_video(vi, pr, fps, gray=False, max_frames=max_frames)
    log.info("wrote %s (%d frames) and %s (%d frames) @ %.1f fps", pt, nt, pr, nr, fps)
    return {"thermal": pt, "rgb": pr}


def _kaist_layout(set_name: str, seq: str) -> str:
    """The public mirror stores some sets per sequence (setXX/V000/lwir) and some flattened (setXX/lwir)."""
    from huggingface_hub import list_repo_tree

    root = f"kaist_train/{set_name}"
    entries = {e.path.rsplit("/", 1)[-1] for e in list_repo_tree(KAIST_REPO, repo_type="dataset", path_in_repo=root)}
    if {"lwir", "visible"} <= entries:
        return root
    if seq in entries:
        sub = {e.path.rsplit("/", 1)[-1] for e in list_repo_tree(KAIST_REPO, repo_type="dataset", path_in_repo=f"{root}/{seq}")}
        if {"lwir", "visible"} <= sub:
            return f"{root}/{seq}"
    raise SystemExit(f"KAIST {set_name}/{seq} is not available in the mirror (both lwir/ and visible/ are needed). "
                     "Known complete: set00 V000/V001/V007/V008 (day), set05 (night).")


def fetch_kaist(out_dir: str, set_name: str = "set05", seq: str = "V000", max_frames: int = 0,
                keep_frames: bool = False, token: Optional[str] = None, start_frame: int = 0) -> Dict[str, str]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:  # pragma: no cover
        raise SystemExit("pip install huggingface_hub  (or: pip install -e '.[data]')") from e
    if set_name not in KAIST_SETS:
        raise SystemExit(f"unknown KAIST set {set_name}; choose from {', '.join(KAIST_SETS)}")
    raw = os.path.join(out_dir, "_frames")
    try:
        sub = _kaist_layout(set_name, seq)
        if max_frames:
            names = [f"I{i:05d}.jpg" for i in range(start_frame, start_frame + max_frames)]
            patterns = [f"{sub}/{m}/{n}" for m in ("lwir", "visible") for n in names]
        else:
            patterns = [f"{sub}/lwir/*", f"{sub}/visible/*"]
        log.info("downloading KAIST %s (%s) from huggingface.co/datasets/%s ...", sub, KAIST_SETS[set_name], KAIST_REPO)
        snapshot_download(repo_id=KAIST_REPO, repo_type="dataset", local_dir=raw, token=token,
                          allow_patterns=patterns, max_workers=16)
    except SystemExit:
        raise
    except Exception as e:
        raise SystemExit(f"could not download from huggingface.co ({type(e).__name__}: {str(e)[:120]}). "
                         "Check your network/proxy, or download the KAIST frames manually and run "
                         "'dualsight frames-to-video --thermal-dir .../lwir --rgb-dir .../visible --out data/kaist'.")
    paths = pair_folders(os.path.join(raw, sub, "lwir"), os.path.join(raw, sub, "visible"), out_dir, KAIST_FPS)
    with open(os.path.join(out_dir, "SOURCE.txt"), "w") as f:
        f.write(f"KAIST Multispectral Pedestrian Benchmark - {sub} ({KAIST_SETS[set_name]})"
                f"{f', frames {start_frame}-{start_frame + max_frames - 1}' if max_frames else ''}\n"
                "S. Hwang, J. Park, N. Kim, Y. Choi, I. S. Kweon, 'Multispectral Pedestrian Detection:\n"
                "Benchmark Dataset and Baseline', CVPR 2015. https://soonminhwang.github.io/rgbt-ped-detection/\n"
                "License: CC BY-NC-SA 4.0 (research / non-commercial use). Do not redistribute.\n")
    if not keep_frames:
        import shutil

        shutil.rmtree(raw, ignore_errors=True)
    return paths


# --------------------------------------------------------------------------------------
# VOT-RGBT (sequences of RGBT234): static surveillance cameras, real fog / night / rain
# --------------------------------------------------------------------------------------

VOT_META = "https://data.votchallenge.net/vot2020/rgbtir/meta/description.json"
VOT_SHOWCASE = {
    "fog6": "fog · pedestrians, umbrella, scooter on a campus road",
    "fog": "dense haze · pedestrians crossing a plaza at long range",
    "walkingnight": "night · groups of pedestrians and a parked/moving car",
    "crossroad": "day · intersection with cars, cyclists and pedestrians",
    "afterrain": "after rain · wet slope, distant pedestrian",
    "carLight": "night · car headlights saturating the visible camera",
}
VOT_FPS = 20.0  # RGBT234 sequences are image series; played back at this rate


def fetch_vot_rgbt(out_dir: str, sequences: List[str], fps: float = VOT_FPS) -> Dict[str, Dict[str, str]]:
    """Download aligned colour + thermal sequences of the VOT-RGBT challenge (RGBT234 subset)."""
    import io
    import json
    import shutil
    import urllib.request
    import zipfile
    from urllib.parse import urljoin

    try:
        with urllib.request.urlopen(VOT_META, timeout=60) as r:
            meta = json.load(r)
    except Exception as e:
        raise SystemExit(f"could not reach data.votchallenge.net ({type(e).__name__}: {str(e)[:120]})")
    by_name = {s["name"]: s for s in meta["sequences"]}
    out: Dict[str, Dict[str, str]] = {}
    for name in sequences:
        if name not in by_name:
            raise SystemExit(f"unknown VOT-RGBT sequence {name}; e.g. {', '.join(VOT_SHOWCASE)}")
        s = by_name[name]
        seq_dir = os.path.join(out_dir, name)
        raw = os.path.join(seq_dir, "_frames")
        for ch in ("color", "ir"):
            url = urljoin(VOT_META, s["channels"][ch]["url"])
            log.info("downloading %s/%s ...", name, ch)
            with urllib.request.urlopen(url, timeout=300) as r:
                zipfile.ZipFile(io.BytesIO(r.read())).extractall(os.path.join(raw, ch))
        paths = pair_folders(os.path.join(raw, "ir"), os.path.join(raw, "color"), seq_dir, fps, pattern="*")
        with open(os.path.join(seq_dir, "SOURCE.txt"), "w") as f:
            f.write(f"VOT-RGBT challenge sequence '{name}' ({VOT_SHOWCASE.get(name, '')}), from RGBT234:\n"
                    "C. Li, X. Liang, Y. Lu, N. Zhao, J. Tang, 'RGB-T object tracking: Benchmark and baseline',\n"
                    "Pattern Recognition 2019; VOT challenge https://www.votchallenge.net/.\n"
                    "Research use only. Do not redistribute the footage.\n")
        shutil.rmtree(raw, ignore_errors=True)
        out[name] = paths
    return out
