"""Sequence-level (temporal) evaluation on CVC-ClinicDB.

Each of the 29 CVC sequences is replayed in frame order as a short clip. The segmentation runs
once; the tracker is then replayed with different ``min_hits`` values to measure what temporal
debouncing buys (fewer false alarms) and what it costs (latency, missed frames).

Caveat: CVC frames were hand-picked from the source videos, so consecutive frames are further
apart in time than at 25 FPS. "Per minute" figures use a nominal 25 FPS and are a relative
comparison between settings, not a clinical false-alarm rate.

    python -m polypseg.video_eval --ckpt runs/unet_r34/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .data import cvc_samples
from .inference import load_segmenter
from .metrics import false_alarms_per_minute, remove_small
from .preprocess import read_image, read_mask
from .tracking import Box, IoUTracker, box_iou, mask_to_boxes


def _hits_gt(box: Box, gt_boxes: list[Box], thr: float) -> bool:
    return any(box_iou(box, g) >= thr for g in gt_boxes)


def replay(
    frames: list[tuple[list[Box], list[Box]]], min_hits: int, max_misses: int, iou_thr: float
) -> dict:
    """frames: per frame (predicted boxes, ground-truth boxes). Returns event statistics."""
    tracker = IoUTracker(min_hits=min_hits, max_misses=max_misses)
    frames_with_gt = frames_alarmed = 0
    first_true_alarm = None
    for i, (pred, gt) in enumerate(frames):
        tracker.update(pred, i)
        shown = tracker.confirmed_tracks()
        if gt:
            frames_with_gt += 1
            if any(_hits_gt(t.box, gt, iou_thr) for t in shown):
                frames_alarmed += 1
                if first_true_alarm is None:
                    first_true_alarm = i
    alarms = [t for t in tracker.finalize() if t.confirmed]
    false_alarms = sum(
        not any(_hits_gt(b, frames[f][1], iou_thr) for f, b in t.history) for t in alarms
    )
    return {
        "frames": len(frames),
        "frames_with_gt": frames_with_gt,
        "frames_alarmed": frames_alarmed,
        "alarms": len(alarms),
        "false_alarms": false_alarms,
        "latency_frames": first_true_alarm,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--min-hits", type=int, nargs="+", default=[1, 2, 3, 5])
    ap.add_argument("--max-misses", type=int, default=2)
    ap.add_argument("--fps", type=float, default=25.0, help="nominal frame rate for per-minute rates")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = torch.load(args.ckpt, map_location="cpu", weights_only=False)["config"]
    ev = cfg["eval"]
    seg = load_segmenter(args.ckpt, args.device)
    sequences: dict[str, list[tuple[list[Box], list[Box]]]] = {}
    for s in tqdm(cvc_samples(cfg["data"]["cvc_root"]), desc="segment", leave=False):
        rgb, gt = read_image(s.image_path), read_mask(s.mask_path)
        min_area = max(1, int(ev["min_component_area_frac"] * gt.size))
        pred = remove_small(seg(rgb) > ev["threshold"], min_area)
        sequences.setdefault(s.group, []).append(
            (mask_to_boxes(pred, min_area), mask_to_boxes(gt, 1))
        )

    rows = []
    for k in args.min_hits:
        per_seq = [replay(f, k, args.max_misses, ev["polyp_iou_threshold"]) for f in sequences.values()]
        n_frames = sum(r["frames"] for r in per_seq)
        lat = [r["latency_frames"] for r in per_seq if r["latency_frames"] is not None]
        rows.append({
            "min_hits": k,
            "sequences_detected": f"{len(lat)}/{len(per_seq)}",
            "frame_alarm_sensitivity": sum(r["frames_alarmed"] for r in per_seq)
            / max(1, sum(r["frames_with_gt"] for r in per_seq)),
            "alarms": sum(r["alarms"] for r in per_seq),
            "false_alarms": sum(r["false_alarms"] for r in per_seq),
            "false_alarms_per_min@nominal_fps": false_alarms_per_minute(
                sum(r["false_alarms"] for r in per_seq), n_frames, args.fps
            ),
            "median_latency_frames": float(np.median(lat)) if lat else None,
        })  # fmt: skip
    df = pd.DataFrame(rows)
    out = Path(args.out or f"outputs/{cfg['run_name']}/cvc_temporal")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "temporal.csv", index=False)
    print(df.round(3).to_string(index=False))
    (out / "temporal.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
