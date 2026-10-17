"""False alarms per minute on colonoscopy videos that contain no polyp, with an ablation of
each pipeline stage (overlay & field-of-view mask, quality gate, temporal confirmation).

Segmentation runs once per frame; per-frame boxes are cached and the tracker is replayed for
every configuration, so the ablation is cheap and every row sees identical model outputs.

    python -m polypseg.false_alarms --model runs/unet_r34/best.onnx \
        --videos data/raw/hyperkvasir-videos/normal
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .inference import load_segmenter
from .metrics import false_alarms_per_minute, remove_small
from .overlays import ignore_mask
from .quality import QualityFilter
from .tracking import Box, IoUTracker, mask_to_boxes
from .video import probe, read_frames


@dataclass
class CachedFrame:
    boxes_raw: list[Box]
    boxes_masked: list[Box]  # after removing device overlays and the area outside the FOV
    low_quality: bool


@dataclass(frozen=True)
class Config:
    name: str
    overlay_mask: bool
    quality_gate: bool
    min_hits: int


ABLATION = (
    Config("raw per-frame output", False, False, 1),
    Config("+ overlay & field-of-view mask", True, False, 1),
    Config("+ quality gate", True, True, 1),
    Config("+ temporal confirmation (3 frames)", True, True, 3),
    Config("  stricter confirmation (5 frames)", True, True, 5),
)


def cache_video(path: Path, segmenter, qf: QualityFilter, thr: float, min_area_frac: float):
    frames = []
    for rgb in read_frames(path):
        prob = segmenter(rgb)
        min_area = max(1, int(min_area_frac * prob.size))
        raw = remove_small(prob > thr, min_area)
        overlay = ignore_mask(rgb)
        masked = raw.copy()
        masked[overlay] = 0
        frames.append(CachedFrame(
            mask_to_boxes(raw, min_area), mask_to_boxes(masked, min_area), not qf.assess(rgb).ok
        ))  # fmt: skip
    return frames


def replay(frames: list[CachedFrame], cfg: Config, max_misses: int) -> tuple[int, int]:
    """Returns (alarms, frames on which a confirmed box is displayed). Every alarm is false."""
    tracker = IoUTracker(min_hits=cfg.min_hits, max_misses=max_misses)
    shown = 0
    for i, f in enumerate(frames):
        if cfg.quality_gate and f.low_quality:
            continue
        tracker.update(f.boxes_masked if cfg.overlay_mask else f.boxes_raw, i)
        shown += bool(tracker.confirmed_tracks())
    return sum(t.confirmed for t in tracker.finalize()), shown


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--videos", required=True, help="folder of polyp-free videos")
    ap.add_argument("--device", default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-area-frac", type=float, default=0.001)
    ap.add_argument("--max-misses", type=int, default=5)
    ap.add_argument("--out", default="outputs/false_alarms")
    args = ap.parse_args()

    seg = load_segmenter(args.model, args.device)
    qf = QualityFilter()
    videos = sorted(p for p in Path(args.videos).iterdir() if p.suffix.lower() in {".avi", ".mp4"})
    cache, total_frames, fps = {}, 0, None
    for v in tqdm(videos, desc="videos"):
        info = probe(v)
        fps = fps or info.fps
        cache[v.name] = cache_video(v, seg, qf, args.threshold, args.min_area_frac)
        total_frames += len(cache[v.name])

    rows = []
    for cfg in ABLATION:
        alarms = shown = 0
        for frames in cache.values():
            a, s = replay(frames, cfg, args.max_misses)
            alarms, shown = alarms + a, shown + s
        rows.append({
            "pipeline": cfg.name,
            "false_alarms": alarms,
            "false_alarms_per_min": round(false_alarms_per_minute(alarms, total_frames, fps), 2),
            "frames_with_false_box_pct": round(100 * shown / total_frames, 2),
        })  # fmt: skip
    low_q = sum(f.low_quality for fr in cache.values() for f in fr)
    meta = {"videos": len(videos), "frames": total_frames, "minutes": round(total_frames / fps / 60, 2),
            "frames_dropped_by_quality_gate_pct": round(100 * low_q / total_frames, 1)}  # fmt: skip
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "ablation.csv", index=False)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
