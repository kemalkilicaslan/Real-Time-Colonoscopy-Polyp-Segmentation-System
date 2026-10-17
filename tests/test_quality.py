import cv2
import numpy as np

from polypseg.quality import QualityFilter, laplacian_variance, specular_ratio


def textured(seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (288, 384, 3), dtype=np.uint8)
    return cv2.resize(cv2.resize(img, (96, 72)), (384, 288), interpolation=cv2.INTER_NEAREST)


def test_blur_lowers_laplacian_variance():
    sharp = textured()
    blurry = cv2.GaussianBlur(sharp, (21, 21), 8)
    assert laplacian_variance(blurry) < 0.2 * laplacian_variance(sharp)


def test_laplacian_variance_is_resolution_invariant():
    img = textured()
    big = cv2.resize(img, (768, 576), interpolation=cv2.INTER_NEAREST)
    a, b = laplacian_variance(img), laplacian_variance(big)
    assert abs(a - b) / a < 0.3


def test_specular_ratio_detects_white_spots():
    img = np.full((100, 100, 3), (180, 60, 60), np.uint8)  # reddish mucosa
    assert specular_ratio(img) == 0.0
    img[:10, :] = 255
    assert specular_ratio(img) == 0.1
    region = np.zeros((100, 100), bool)
    region[50:] = True
    assert specular_ratio(img, region) == 0.0


def test_quality_filter_flags_blur_and_darkness():
    qf = QualityFilter(min_blur=50)
    assert qf.assess(textured()).ok
    assert "blur" in qf.assess(cv2.GaussianBlur(textured(), (31, 31), 12)).reasons
    assert "dark" in qf.assess(np.zeros((100, 100, 3), np.uint8)).reasons


def test_blur_score_ignores_sharp_device_overlays_and_text():
    """Regression: a burnt-in ScopeGuide box and text made every video frame look sharp."""
    frame = np.zeros((576, 720, 3), np.uint8)
    cv2.circle(frame, (360, 288), 270, (200, 90, 70), -1)  # field of view
    blurry = cv2.GaussianBlur(frame, (1, 1), 0)
    blurry[60:80, 40:200] = 0
    cv2.putText(blurry, "11/10/2012 CVP:1", (40, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,) * 3, 1)
    blurry[380:540, 30:200] = (20, 165, 137)  # ScopeGuide box overlapping the FOV
    assert laplacian_variance(blurry) < QualityFilter().min_blur
