import math

import numpy as np
import pytest

from polypseg.metrics import (
    bootstrap_ci,
    confusion,
    dice,
    false_alarms_per_minute,
    iou,
    polyp_level,
    remove_small,
    segmentation_metrics,
)


def square(h, w, y0, x0, size):
    m = np.zeros((h, w), np.uint8)
    m[y0 : y0 + size, x0 : x0 + size] = 1
    return m


def test_confusion_counts():
    gt = np.array([[1, 1], [0, 0]])
    pred = np.array([[1, 0], [1, 0]])
    assert confusion(pred, gt) == (1, 1, 1, 1)


def test_perfect_prediction():
    gt = square(20, 20, 5, 5, 6)
    m = segmentation_metrics(gt, gt)
    assert m["dice"] == m["iou"] == m["sensitivity"] == m["specificity"] == m["precision"] == 1.0


def test_known_overlap_values():
    gt = np.zeros((10, 10), np.uint8)
    gt[:, :4] = 1  # 40 px
    pred = np.zeros_like(gt)
    pred[:, 2:6] = 1  # 40 px, 20 overlap
    assert dice(pred, gt) == pytest.approx(2 * 20 / 80)
    assert iou(pred, gt) == pytest.approx(20 / 60)
    m = segmentation_metrics(pred, gt)
    assert m["sensitivity"] == pytest.approx(0.5)
    assert m["specificity"] == pytest.approx(40 / 60)


def test_empty_conventions():
    empty = np.zeros((8, 8), np.uint8)
    m = segmentation_metrics(empty, empty)
    assert m["dice"] == 1.0 and m["iou"] == 1.0
    assert math.isnan(m["sensitivity"]) and math.isnan(m["precision"])
    assert dice(square(8, 8, 0, 0, 2), empty) == 0.0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        confusion(np.zeros((4, 4)), np.zeros((4, 5)))


def test_polyp_level_detection_and_false_positive():
    gt = square(100, 100, 10, 10, 20) | square(100, 100, 60, 60, 20)
    pred = square(100, 100, 12, 12, 20) | square(100, 100, 5, 70, 10)  # hits polyp 1, + FP blob
    r = polyp_level(pred, gt, min_iou=0.1)
    assert r == {"n_gt_polyps": 2, "n_detected": 1, "n_pred_blobs": 2, "n_false_pos_blobs": 1}


def test_polyp_level_min_area_filters_specks():
    gt = square(50, 50, 10, 10, 10)
    pred = gt.copy()
    pred[40, 40] = 1  # 1-px speck
    assert polyp_level(pred, gt, min_area=5)["n_false_pos_blobs"] == 0
    assert polyp_level(pred, gt, min_area=1)["n_false_pos_blobs"] == 1


def test_remove_small():
    m = square(30, 30, 0, 0, 5)
    m[20, 20] = 1
    assert remove_small(m, 4).sum() == 25


def test_false_alarms_per_minute():
    assert false_alarms_per_minute(3, 25 * 60, 25.0) == pytest.approx(3.0)
    with pytest.raises(ValueError):
        false_alarms_per_minute(1, 0, 25.0)


def test_bootstrap_ci_contains_mean_and_ignores_nan():
    v = np.r_[np.linspace(0.6, 0.9, 50), np.nan]
    lo, hi = bootstrap_ci(v, n_boot=500)
    assert lo < np.nanmean(v) < hi
