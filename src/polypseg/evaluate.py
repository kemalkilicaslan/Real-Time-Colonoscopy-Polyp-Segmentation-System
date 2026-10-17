"""Image-level evaluation with pixel- and polyp-level metrics.

    # in-domain: held-out Kvasir-SEG validation split
    python -m polypseg.evaluate --ckpt runs/unet_r34/best.pt --dataset kvasir-val
    # out-of-domain: every CVC-ClinicDB frame (different hospital, scope and resolution)
    python -m polypseg.evaluate --ckpt runs/unet_r34/best.pt --dataset cvc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .data import Sample, cvc_samples, kvasir_samples, negative_samples
from .inference import load_segmenter
from .metrics import bootstrap_ci, polyp_level, remove_small, segmentation_metrics
from .model import load_config
from .preprocess import read_image, read_mask

PIXEL_METRICS = ["dice", "iou", "sensitivity", "specificity", "precision"]


def resolve_samples(dataset: str, cfg: dict, ckpt: Path) -> list[Sample]:
    d = cfg["data"]
    if dataset == "cvc":
        return cvc_samples(d["cvc_root"])
    split = json.loads((ckpt.parent / "split.json").read_text())
    if dataset == "kvasir-val":
        return [s for s in kvasir_samples(d["kvasir_root"]) if s.name in set(split["val"])]
    if dataset == "negatives-val":
        # Held-out polyp-free frames. Models trained without negatives have no "neg_val" list,
        # so they are scored on the same held-out frames as the model that saw negatives.
        root = d.get("negatives_root", "data/raw/hyperkvasir-negatives")
        names = split.get("neg_val")
        if names is None:
            ref = Path("runs") / "unet_r34_neg" / "split.json"
            names = json.loads(ref.read_text())["neg_val"]
        return [s for s in negative_samples(root) if s.name in set(names)]
    raise ValueError(f"unknown dataset {dataset!r}")


def evaluate(segmenter, samples: list[Sample], ev: dict, pred_dir: Path | None) -> pd.DataFrame:
    rows = []
    for s in tqdm(samples, desc="eval", leave=False):
        rgb = read_image(s.image_path)
        gt = (
            read_mask(s.mask_path)
            if s.mask_path is not None
            else np.zeros(rgb.shape[:2], dtype=np.uint8)
        )
        prob = segmenter(rgb)
        min_area = max(1, int(ev["min_component_area_frac"] * gt.size))
        pred = remove_small(prob > ev["threshold"], min_area)
        row = {"name": s.name, "group": s.group, "gt_area_frac": float(gt.mean())}
        row.update(segmentation_metrics(pred, gt))
        row.update(polyp_level(pred, gt, ev["polyp_iou_threshold"], min_area))
        rows.append(row)
        if pred_dir is not None:
            cv2.imwrite(str(pred_dir / f"{s.name}.png"), pred * 255)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    out: dict = {"n_images": len(df)}
    for k in PIXEL_METRICS:
        out[k] = float(np.nanmean(df[k]))
    for k in ["dice", "iou"]:
        out[f"{k}_ci95"] = [round(b, 4) for b in bootstrap_ci(df[k].to_numpy())]
    out["polyp_sensitivity"] = float(df["n_detected"].sum() / max(1, df["n_gt_polyps"].sum()))
    out["false_pos_blobs_per_image"] = float(df["n_false_pos_blobs"].mean())
    out["images_with_false_pos"] = float((df["n_false_pos_blobs"] > 0).mean())
    out["failure_rate_dice_lt_0.5"] = float((df["dice"] < 0.5).mean())
    if df["group"].nunique() < len(df):  # grouped data (CVC sequences)
        per_group = df.groupby("group")["dice"].mean()
        out["n_groups"] = int(per_group.size)
        out["group_dice_min"] = float(per_group.min())
        out["group_dice_median"] = float(per_group.median())
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ckpt", required=True, help=".pt checkpoint (or .onnx with --config)")
    ap.add_argument("--config", default=None, help="config yaml; defaults to the checkpoint's own")
    ap.add_argument("--dataset", choices=["kvasir-val", "cvc", "negatives-val"], required=True)
    ap.add_argument("--out", default=None, help="default: outputs/<run_name>/<dataset>")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    if args.config:
        cfg = load_config(args.config)
    else:
        import torch

        cfg = torch.load(ckpt, map_location="cpu", weights_only=False)["config"]
    out = Path(args.out or f"outputs/{cfg['run_name']}/{args.dataset}")
    (out / "pred").mkdir(parents=True, exist_ok=True)

    samples = resolve_samples(args.dataset, cfg, ckpt)
    df = evaluate(load_segmenter(ckpt, args.device), samples, cfg["eval"], out / "pred")
    df.to_csv(out / "per_image.csv", index=False)
    if df["group"].nunique() < len(df):
        df.groupby("group")[["dice", "iou", "sensitivity"]].mean().round(4).to_csv(
            out / "per_group.csv"
        )
    summary = summarize(df)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
