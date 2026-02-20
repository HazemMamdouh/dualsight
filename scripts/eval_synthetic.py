#!/usr/bin/env python3
"""Score a run on the synthetic demo against its ground truth (scenario.json).

    python scripts/eval_synthetic.py --run runs/demo --gt data/demo/scenario.json

Per actor: was it turned into an event, with the right category, how fast, and for how
much of its visible time. Plus: false (unmatched) events and gate accuracy (fog / camera motion).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd


def inside(pt, box, pad=12):
    return box[0] - pad <= pt[0] <= box[2] + pad and box[1] - pad <= pt[1] <= box[3] + pad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--md", action="store_true", help="print a markdown table")
    a = ap.parse_args()

    gt = json.load(open(a.gt))
    fps = gt["fps"]
    W, H = gt["rgb_size"]
    events = json.load(open(f"{a.run}/events.json"))
    frames = pd.read_csv(f"{a.run}/frames.csv")

    vis = defaultdict(dict)  # actor -> frame -> box (clipped, at least 30 % inside the frame)
    for fr in gt["frames"]:
        for o in fr["objects"]:
            x1, y1, x2, y2 = o["rgb_box"]
            cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
            if cx2 <= cx1 or cy2 <= cy1:
                continue
            if (cx2 - cx1) * (cy2 - cy1) >= 0.3 * (x2 - x1) * (y2 - y1):
                vis[o["id"]][fr["frame"]] = [cx1, cy1, cx2, cy2]

    # assign events to actors by trajectory hits
    hits = defaultdict(lambda: defaultdict(int))
    for e in events:
        for x, y, f in e["path"]:
            for aid, boxes in vis.items():
                b = boxes.get(int(f))
                if b is not None and inside((x, y), b):
                    hits[e["id"]][aid] += 1
    owner = {eid: max(h, key=h.get) for eid, h in hits.items() if h}
    rows = []
    for aid, act in enumerate(gt["actors"]):
        fs = sorted(vis.get(aid, {}))
        if not fs:
            continue
        evs = [e for e in events if owner.get(e["id"]) == aid]
        t_vis = fs[0] / fps
        if evs:
            first = min(e["start_s"] for e in evs)
            covered = set()
            for e in evs:
                covered |= set(range(e["start_frame"], e["end_frame"] + 1))
            cov = len(covered & set(fs)) / len(fs)
            cats = sorted({e["category"] for e in evs})
            ok = act["kind"] in cats
            cls = ", ".join(sorted({e["class"] for e in evs if e["class"]}))
        else:
            first, cov, cats, ok, cls = None, 0.0, [], False, ""
        rows.append({"actor": f"{act['kind']}#{aid}", "visible": f"{t_vis:.1f}-{fs[-1] / fps:.1f}s",
                     "detected": bool(evs), "category": "/".join(cats) or "-", "classes": cls or "-",
                     "correct": ok, "latency_s": None if first is None else max(0.0, round(first - t_vis, 2)),
                     "coverage": round(cov, 2), "events": len(evs)})
    false_ev = [e for e in events if e["id"] not in owner]

    # gate accuracy
    gfog = np.array([fr["fog"] > 0.5 for fr in gt["frames"]][: len(frames)])
    gcam = np.array([fr["camera_moving"] for fr in gt["frames"]][: len(frames)])
    fog_acc = float((frames["fog_on"].to_numpy()[: len(gfog)].astype(bool) == gfog).mean())
    cam_acc = float((frames["deblur_on"].to_numpy()[: len(gcam)].astype(bool) == gcam).mean())

    df = pd.DataFrame(rows)
    if a.md:
        print(df.to_markdown(index=False))
    else:
        print(df.to_string(index=False))
    print(f"\nactors detected: {int(df['detected'].sum())}/{len(df)} | correct category: {int(df['correct'].sum())}/{len(df)}"
          f" | false events: {len(false_ev)} | fog-gate agreement: {fog_acc:.1%} | camera-gate agreement: {cam_acc:.1%}")
    for e in false_ev:
        print(f"  unmatched event #{e['id']} {e['category']} {e['start_s']}-{e['end_s']}s box={e['max_box']}")


if __name__ == "__main__":
    main()
