"""Pixel-, polyp- and video-level metrics.

Conventions for degenerate cases are explicit because they change the reported numbers:
* Dice / IoU are 1.0 when both prediction and ground truth are empty, 0.0 when only one is.
* Sensitivity is NaN when the ground truth is empty; precision is NaN when nothing is predicted.
  NaNs are skipped when averaging (``np.nanmean``).
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np


def confusion(pred: np.ndarray, gt: np.ndarray) -> tuple[int, int, int, int]:
    """Return (tp, fp, fn, tn) for two binary masks of the same shape."""
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    p, g = pred.astype(bool), gt.astype(bool)
    tp = int(np.count_nonzero(p & g))
    fp = int(np.count_nonzero(p & ~g))
    fn = int(np.count_nonzero(~p & g))
    tn = int(p.size - tp - fp - fn)
    return tp, fp, fn, tn


def _ratio(num: float, den: float, empty: float) -> float:
    return float(num / den) if den > 0 else empty


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    tp, fp, fn, _ = confusion(pred, gt)
    return _ratio(2 * tp, 2 * tp + fp + fn, 1.0)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    tp, fp, fn, _ = confusion(pred, gt)
    return _ratio(tp, tp + fp + fn, 1.0)


def segmentation_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    tp, fp, fn, tn = confusion(pred, gt)
    return {
        "dice": _ratio(2 * tp, 2 * tp + fp + fn, 1.0),
        "iou": _ratio(tp, tp + fp + fn, 1.0),
        "sensitivity": _ratio(tp, tp + fn, float("nan")),
        "specificity": _ratio(tn, tn + fp, float("nan")),
        "precision": _ratio(tp, tp + fp, float("nan")),
    }


def components(mask: np.ndarray, min_area: int = 1) -> list[np.ndarray]:
    """Split a binary mask into connected components (8-connectivity), dropping tiny blobs."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [labels == i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]


def polyp_level(
    pred: np.ndarray, gt: np.ndarray, min_iou: float = 0.1, min_area: int = 1
) -> dict[str, int]:
    """Lesion-level matching: what a clinician sees is "was this polyp flagged", not pixel overlap.

    A ground-truth polyp is *detected* if some predicted blob overlaps it with IoU >= ``min_iou``.
    A predicted blob that matches no ground-truth polyp is a *false positive*.
    """
    gt_comps = components(gt, 1)
    pred_comps = components(pred, min_area)
    ious = np.zeros((len(gt_comps), len(pred_comps)), dtype=np.float32)
    for i, g in enumerate(gt_comps):
        for j, p in enumerate(pred_comps):
            ious[i, j] = iou(p, g)
    hit = ious >= min_iou
    return {
        "n_gt_polyps": len(gt_comps),
        "n_detected": int(hit.any(axis=1).sum()) if gt_comps else 0,
        "n_pred_blobs": len(pred_comps),
        "n_false_pos_blobs": int((~hit.any(axis=0)).sum()) if pred_comps else 0,
    }


def remove_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=np.uint8)
    for comp in components(mask, min_area):
        out[comp] = 1
    return out


def false_alarms_per_minute(n_false_alarms: int, n_frames: int, fps: float) -> float:
    if n_frames <= 0 or fps <= 0:
        raise ValueError("n_frames and fps must be positive")
    return n_false_alarms / (n_frames / fps / 60.0)


def bootstrap_ci(
    values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of the mean (NaNs dropped)."""
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
