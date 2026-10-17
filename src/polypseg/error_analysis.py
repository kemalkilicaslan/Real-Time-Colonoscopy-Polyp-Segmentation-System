"""Group failures by image attribute and render a gallery per group.

Run after ``polypseg.evaluate`` (it reads ``per_image.csv`` and the saved ``pred/`` masks):

    python -m polypseg.error_analysis --eval-dir outputs/unet_r34/cvc

Attributes are measured from the image and ground truth, not hand-labelled:
* small        polyp covers < 3% of the frame
* specular     > 5% of the polyp region (dilated) is specular highlight
* blurry       Laplacian variance in the bottom quartile of the dataset
* low_contrast colour difference (CIEDE-like, Lab) between polyp and surrounding ring in the
               bottom quartile; a proxy for flat / sessile lesions, since neither dataset
               provides Paris-classification morphology labels
* multiple     more than one polyp in the frame
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .data import cvc_samples, kvasir_samples
from .metrics import components
from .preprocess import read_image, read_mask
from .quality import laplacian_variance, specular_ratio

FAIL_DICE = 0.5
SMALL_FRAC = 0.03
SPECULAR_FRAC = 0.05


def contrast(rgb: np.ndarray, gt: np.ndarray) -> float:
    """Mean Lab colour distance between the polyp and a ring of surrounding mucosa."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    ring = (cv2.dilate(gt, k) > 0) & (gt == 0)
    inside = gt > 0
    if not ring.any() or not inside.any():
        return float("nan")
    return float(np.linalg.norm(lab[inside].mean(0) - lab[ring].mean(0)))


def attributes(rgb: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    dil = cv2.dilate(gt, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    return {
        "blur_score": laplacian_variance(rgb),
        "specular_in_polyp": specular_ratio(rgb, dil),
        "contrast": contrast(rgb, gt),
        "n_polyps": len(components(gt, 1)),
    }


def overlay(rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """GT contour in green, prediction filled in magenta."""
    out = rgb.copy()
    out[pred > 0] = (0.5 * out[pred > 0] + 0.5 * np.array([255, 0, 200])).astype(np.uint8)
    cnts, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (0, 255, 0), 2)
    return out


def gallery(tiles: list[tuple[np.ndarray, str]], path: Path, cols: int = 4, w: int = 320) -> None:
    cell_h = int(w * 0.75)
    cells = []
    for img, label in tiles:
        scale = min(w / img.shape[1], cell_h / img.shape[0])
        resized = cv2.resize(img, (round(img.shape[1] * scale), round(img.shape[0] * scale)))
        cell = np.zeros((cell_h, w, 3), np.uint8)  # letterbox, keep aspect ratio
        y, x = (cell_h - resized.shape[0]) // 2, (w - resized.shape[1]) // 2
        cell[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        cv2.rectangle(cell, (0, 0), (w, 22), (0, 0, 0), -1)
        cv2.putText(cell, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)  # fmt: skip
        cells.append(cell)
    while len(cells) % cols:
        cells.append(np.zeros_like(cells[0]))
    rows = [np.hstack(cells[i : i + cols]) for i in range(0, len(cells), cols)]
    cv2.imwrite(str(path), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--eval-dir", required=True, help="output dir of polypseg.evaluate")
    ap.add_argument("--ckpt", default=None, help="checkpoint, to locate the data (default: guess)")
    ap.add_argument("--per-group", type=int, default=8, help="gallery size per attribute")
    args = ap.parse_args()

    ev = Path(args.eval_dir)
    df = pd.read_csv(ev / "per_image.csv")
    ckpt = Path(args.ckpt or Path("runs") / ev.parent.name / "best.pt")
    cfg = torch.load(ckpt, map_location="cpu", weights_only=False)["config"]
    samples = (
        cvc_samples(cfg["data"]["cvc_root"]) if ev.name == "cvc"
        else kvasir_samples(cfg["data"]["kvasir_root"])
    )  # fmt: skip
    by_name = {s.name: s for s in samples}

    attrs = []
    for name in tqdm(df["name"], desc="attributes", leave=False):
        s = by_name[name]
        attrs.append(attributes(read_image(s.image_path), read_mask(s.mask_path)))
    df = pd.concat([df, pd.DataFrame(attrs)], axis=1)

    df["small"] = df["gt_area_frac"] < SMALL_FRAC
    df["specular"] = df["specular_in_polyp"] > SPECULAR_FRAC
    df["blurry"] = df["blur_score"] < df["blur_score"].quantile(0.25)
    df["low_contrast"] = df["contrast"] < df["contrast"].quantile(0.25)
    df["multiple"] = df["n_polyps"] > 1
    df["failed"] = df["dice"] < FAIL_DICE
    groups = ["small", "specular", "blurry", "low_contrast", "multiple"]

    rows = [{"group": "all", "n": len(df), "failure_rate": df["failed"].mean(),
             "mean_dice": df["dice"].mean(), "share_of_failures": 1.0}]  # fmt: skip
    n_fail = max(1, int(df["failed"].sum()))
    for g in groups:
        sub = df[df[g]]
        if len(sub) == 0:
            continue
        rows.append({
            "group": g,
            "n": len(sub),
            "failure_rate": sub["failed"].mean(),
            "mean_dice": sub["dice"].mean(),
            "share_of_failures": sub["failed"].sum() / n_fail,
        })  # fmt: skip
    table = pd.DataFrame(rows).round(3)
    out = ev / "error_analysis"
    (out / "gallery").mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "by_attribute.csv", index=False)
    df.to_csv(out / "per_image_with_attributes.csv", index=False)
    print(table.to_string(index=False))

    for g in [*groups, "failed"]:
        worst = df[df[g] & df["failed"]].nsmallest(args.per_group, "dice")
        if worst.empty:
            continue
        tiles = []
        for _, r in worst.iterrows():
            s = by_name[r["name"]]
            pred = cv2.imread(str(ev / "pred" / f"{r['name']}.png"), cv2.IMREAD_GRAYSCALE) > 127
            img = overlay(read_image(s.image_path), read_mask(s.mask_path), pred)
            tiles.append((img, f"{r['name']}  dice {r['dice']:.2f}"))
        gallery(tiles, out / "gallery" / f"{g}.jpg")
    (out / "README.json").write_text(json.dumps({
        "failure_definition": f"dice < {FAIL_DICE}",
        "legend": "green contour = ground truth, magenta fill = prediction",
        "thresholds": {"small_area_frac": SMALL_FRAC, "specular_frac": SPECULAR_FRAC,
                       "blurry": "bottom quartile Laplacian variance",
                       "low_contrast": "bottom quartile Lab distance"},
    }, indent=2))  # fmt: skip


if __name__ == "__main__":
    main()
