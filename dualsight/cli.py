"""DualSight command-line interface.

    dualsight run       --thermal T --rgb R [--device cpu|cuda] [--show] [--save-video] ...
    dualsight render    --run runs/x --thermal T --rgb R [--view enhanced|raw] [--compare]
    dualsight report    --run runs/x
    dualsight make-demo --out data/demo
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

from . import __version__
from .config import load_config, set_dotted

log = logging.getLogger("dualsight")


def _setup_logging(out_dir: str | None, verbose: bool) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    log.handlers.clear()
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(out_dir, "pipeline.log"), mode="a")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    try:  # keep third-party chatter out of the operator log
        from ultralytics.utils import LOGGER as UL

        UL.setLevel(logging.WARNING)
    except Exception:
        pass


def _common_cfg(args) -> dict:
    cfg = load_config(args.config)
    if getattr(args, "device", None):
        cfg["device"] = args.device
    for kv in getattr(args, "set", None) or []:
        if "=" not in kv:
            raise SystemExit(f"--set expects key=value, got {kv}")
        k, v = kv.split("=", 1)
        set_dotted(cfg, k, v)
    return cfg


def cmd_run(args) -> int:
    out = args.out or os.path.join("runs", time.strftime("%Y%m%d-%H%M%S"))
    _setup_logging(out, args.verbose)
    cfg = _common_cfg(args)
    s = cfg["sources"]
    if args.gstreamer is not None:
        s["use_gstreamer"] = args.gstreamer
    if args.hw_decode:
        s["hw_decode"] = args.hw_decode
    if args.realtime:
        s["realtime"] = True
    if args.max_frames:
        s["max_frames"] = args.max_frames
    if args.start:
        s["start_s"] = args.start
    if args.thermal_enhancer:
        cfg["thermal"]["enhancer"] = args.thermal_enhancer
    if args.flow:
        cfg["motion"]["flow"]["backend"] = args.flow
    if args.detector:
        cfg["detector"]["model"] = args.detector
    if args.no_detector:
        cfg["detector"]["enabled"] = False

    from .pipeline import Pipeline

    log.info("DualSight %s | thermal=%s | rgb=%s | out=%s", __version__, args.thermal, args.rgb, out)
    pipe = Pipeline(cfg, out, show=args.show, save_video=args.save_video, save_processed=args.save_processed)
    pipe.run(args.thermal, args.rgb)
    if not args.no_report:
        from .analytics.report import build_report

        idx = build_report(out)
        log.info("report: %s", idx)
    log.info("outputs in %s", os.path.abspath(out))
    return 0


def cmd_render(args) -> int:
    _setup_logging(None, args.verbose)
    from .render import render

    out = render(args.run, args.thermal, args.rgb, out=args.out, view=args.view, compare=args.compare,
                 show=args.show, with_report=not args.no_report, max_frames=args.max_frames)
    for p in out:
        log.info("wrote %s", p)
    return 0


def cmd_report(args) -> int:
    _setup_logging(None, args.verbose)
    from .analytics.report import build_report

    log.info("report: %s", build_report(args.run, args.out))
    return 0


def cmd_make_demo(args) -> int:
    _setup_logging(None, args.verbose)
    from .synth.scene import generate_demo

    def prog(i, n):
        if i % 60 == 0 or i == n:
            log.info("rendered %d/%d frames", i, n)

    paths = generate_demo(args.out, seconds=args.seconds, fps=args.fps, seg_weights=args.seg_model, progress=prog)
    for k, v in paths.items():
        log.info("%s: %s", k, v)
    return 0


def cmd_fetch_data(args) -> int:
    _setup_logging(None, args.verbose)
    from .synth.real_data import KAIST_SETS, VOT_SHOWCASE, fetch_kaist, fetch_vot_rgbt

    if args.list:
        for k, v in (KAIST_SETS if args.dataset == "kaist" else VOT_SHOWCASE).items():
            print(f"{k:14s} {v}")
        return 0
    if args.dataset == "vot-rgbt":
        seqs = [s for s in (args.seq if args.seq != "V000" else "fog6,walkingnight,crossroad").split(",") if s]
        for name, p in fetch_vot_rgbt(args.out if args.out != "data/kaist" else "data/vot", seqs).items():
            log.info("ready (%s): dualsight run --thermal %s --rgb %s", name, p["thermal"], p["rgb"])
        return 0
    paths = fetch_kaist(args.out, args.set, args.seq, args.max_frames, args.keep_frames, start_frame=args.start_frame)
    log.info("ready: dualsight run --thermal %s --rgb %s", paths["thermal"], paths["rgb"])
    return 0


def cmd_frames_to_video(args) -> int:
    _setup_logging(None, args.verbose)
    from .synth.real_data import pair_folders

    pair_folders(args.thermal_dir, args.rgb_dir, args.out, args.fps, args.max_frames)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="dualsight", description="All-weather thermal + RGB motion intelligence pipeline")
    ap.add_argument("--version", action="version", version=f"dualsight {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the live pipeline on two synced streams")
    r.add_argument("--thermal", required=True, help="thermal source: file, rtsp://, udp://PORT, /dev/videoN or gst:<pipeline>")
    r.add_argument("--rgb", required=True, help="RGB source (same forms as --thermal)")
    r.add_argument("--out", help="run directory (default runs/<timestamp>)")
    r.add_argument("--config", help="YAML config (see configs/default.yaml)")
    r.add_argument("--device", help="auto | cpu | cuda | cuda:N | mps")
    r.add_argument("--show", action="store_true", help="live visualization window (press q to stop)")
    r.add_argument("--save-video", action="store_true", help="save the annotated dashboard stream to live.mp4")
    r.add_argument("--save-processed", action="store_true", help="also save thermal_enhanced.mp4 and rgb_deblurred.mp4")
    r.add_argument("--realtime", action="store_true", help="pace offline files as a live stream (drops frames when behind)")
    g = r.add_mutually_exclusive_group()
    g.add_argument("--gstreamer", dest="gstreamer", action="store_const", const=True, default=None, help="force GStreamer ingest")
    g.add_argument("--no-gstreamer", dest="gstreamer", action="store_const", const=False, help="force OpenCV/FFmpeg ingest")
    r.add_argument("--hw-decode", choices=["none", "nvidia", "jetson"], help="hardware decoder for GStreamer ingest")
    r.add_argument("--thermal-enhancer", choices=["auto", "classical", "esrgan", "hybrid", "off"])
    r.add_argument("--flow", choices=["auto", "dis", "farneback", "cuda_farneback", "nvof", "raft"])
    r.add_argument("--detector", help="Ultralytics detection weights (default yolo11n.pt)")
    r.add_argument("--no-detector", action="store_true", help="motion only, skip object attribution")
    r.add_argument("--max-frames", type=int, default=0)
    r.add_argument("--start", type=float, default=0.0, help="start offset in seconds (files only)")
    r.add_argument("--no-report", action="store_true")
    r.add_argument("--set", action="append", metavar="KEY=VALUE", help="override any config key, e.g. motion.combine=or")
    r.add_argument("-v", "--verbose", action="store_true")
    r.set_defaults(func=cmd_run)

    v = sub.add_parser("render", help="re-create visualizations from a run directory + the input videos")
    v.add_argument("--run", required=True)
    v.add_argument("--thermal", help="thermal input (default: path recorded in run.json)")
    v.add_argument("--rgb", help="RGB input (default: path recorded in run.json)")
    v.add_argument("--out", help="output video (default <run>/render.mp4)")
    v.add_argument("--view", choices=["enhanced", "raw"], default="enhanced",
                   help="enhanced: re-apply the logged enhancement decisions; raw: draw on the untouched inputs")
    v.add_argument("--compare", action="store_true", help="also write a before/after 2x2 comparison video")
    v.add_argument("--show", action="store_true")
    v.add_argument("--max-frames", type=int, default=0)
    v.add_argument("--no-report", action="store_true")
    v.add_argument("-v", "--verbose", action="store_true")
    v.set_defaults(func=cmd_render)

    p = sub.add_parser("report", help="(re)build the HTML/PNG analytics report of a run")
    p.add_argument("--run", required=True)
    p.add_argument("--out")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_report)

    f = sub.add_parser("fetch-data", help="download real synced thermal+RGB footage (KAIST multispectral)")
    f.add_argument("dataset", choices=["vot-rgbt", "kaist"],
                   help="vot-rgbt: static surveillance cameras with real fog/night (recommended); kaist: vehicle-mounted")
    f.add_argument("--set", default="set05", help="set00-02 day, set03-05 night (default set05: night downtown)")
    f.add_argument("--start-frame", type=int, default=0, help="first frame index to fetch (with --max-frames)")
    f.add_argument("--seq", default="V000", help="KAIST sequence, or comma-separated VOT-RGBT names (default fog6,walkingnight,crossroad)")
    f.add_argument("--out", default="data/kaist")
    f.add_argument("--max-frames", type=int, default=0)
    f.add_argument("--keep-frames", action="store_true")
    f.add_argument("--list", action="store_true", help="list available sets")
    f.add_argument("-v", "--verbose", action="store_true")
    f.set_defaults(func=cmd_fetch_data)

    c = sub.add_parser("frames-to-video", help="turn two index-aligned frame folders into synced thermal.mp4 + rgb.mp4")
    c.add_argument("--thermal-dir", required=True)
    c.add_argument("--rgb-dir", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--fps", type=float, default=20.0)
    c.add_argument("--max-frames", type=int, default=0)
    c.add_argument("-v", "--verbose", action="store_true")
    c.set_defaults(func=cmd_frames_to_video)

    d = sub.add_parser("make-demo", help="generate the synthetic, license-clean demo pair")
    d.add_argument("--out", default="data/demo")
    d.add_argument("--seconds", type=float)
    d.add_argument("--fps", type=float)
    d.add_argument("--seg-model", default="yolo11n-seg.pt")
    d.add_argument("-v", "--verbose", action="store_true")
    d.set_defaults(func=cmd_make_demo)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
