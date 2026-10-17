"""Operating-point sweep: sensitivity on CVC-ClinicDB vs false alarms on polyp-free video.

A CADe system is tuned, not just trained: lowering the probability threshold recovers subtle
polyps and costs false alarms. Each frame is segmented once; every threshold reuses the same
probability map, so the rows differ only in the threshold.

    python -m polypseg.operating_point --ckpt runs/unet_r34_neg/best.pt \
        --videos data/raw/hyperkvasir-videos/normal
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from .data import cvc_samples
from .false_alarms import ABLATION, CachedFrame, replay
from .inference import load_segmenter
from .metrics import dice, false_alarms_per_minute, polyp_level, remove_small
from .overlays import ignore_mask
from .preprocess import read_image, read_mask
from .quality import QualityFilter
from .tracking import mask_to_boxes
from .video import probe, read_frames

FULL_PIPELINE = next(c for c in ABLATION if c.min_hits == 3)


def cvc_sweep(seg, cfg: dict, thresholds: list[float]) -> pd.DataFrame:
    ev = cfg["eval"]
    rows = []
    for s in tqdm(cvc_samples(cfg["data"]["cvc_root"]), desc="cvc", leave=False):
        rgb, gt = read_image(s.image_path), read_mask(s.mask_path)
        prob = seg(rgb)
        min_area = max(1, int(ev["min_component_area_frac"] * gt.size))
        for thr in thresholds:
            pred = remove_small(prob > thr, min_area)
            pl = polyp_level(pred, gt, ev["polyp_iou_threshold"], min_area)
            rows.append({"threshold": thr, "group": s.group, "dice": dice(pred, gt), **pl})
    df = pd.DataFrame(rows)
    out = []
    for thr, g in df.groupby("threshold"):
        per_seq = g.groupby("group")["dice"].mean()
        out.append({
            "threshold": thr,
            "cvc_dice": g["dice"].mean(),
            "cvc_polyp_sensitivity": g["n_detected"].sum() / g["n_gt_polyps"].sum(),
            "cvc_worst_sequence_dice": per_seq.min(),
        })  # fmt: skip
    return pd.DataFrame(out)


def video_sweep(seg, videos: list[Path], thresholds: list[float], min_area_frac: float):
    qf = QualityFilter()
    cache: dict[float, list[list[CachedFrame]]] = {t: [] for t in thresholds}
    n_frames, fps = 0, None
    for v in tqdm(videos, desc="videos"):
        fps = fps or probe(v).fps
        per_thr: dict[float, list[CachedFrame]] = {t: [] for t in thresholds}
        for rgb in read_frames(v):
            prob = seg(rgb)
            prob[ignore_mask(rgb)] = 0.0
            low_q = not qf.assess(rgb).ok
            min_area = max(1, int(min_area_frac * prob.size))
            for thr in thresholds:
                boxes = mask_to_boxes(remove_small(prob > thr, min_area), min_area)
                per_thr[thr].append(CachedFrame(boxes, boxes, low_q))
            n_frames += 1
        for thr in thresholds:
            cache[thr].append(per_thr[thr])
    rows = []
    for thr in thresholds:
        alarms = sum(replay(frames, FULL_PIPELINE, 5)[0] for frames in cache[thr])
        rows.append({"threshold": thr,
                     "false_alarms_per_min": false_alarms_per_minute(alarms, n_frames, fps)})  # fmt: skip
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--videos", required=True, help="folder of polyp-free videos")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = torch.load(args.ckpt, map_location="cpu", weights_only=False)["config"]
    seg = load_segmenter(args.ckpt, args.device)
    videos = sorted(p for p in Path(args.videos).iterdir() if p.suffix.lower() in {".avi", ".mp4"})
    df = cvc_sweep(seg, cfg, args.thresholds).merge(
        video_sweep(seg, videos, args.thresholds, cfg["eval"]["min_component_area_frac"]),
        on="threshold",
    )
    out = Path(args.out or f"outputs/{cfg['run_name']}/operating_points.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(out, index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
