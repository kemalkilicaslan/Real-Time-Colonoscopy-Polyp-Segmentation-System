"""Frame quality assessment: blur, specular highlights and exposure.

Colonoscopy frames are often unusable (motion blur, the scope touching the mucosa, flushing).
Running the detector on them mostly produces false alarms, so the video pipeline drops them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


def field_of_view(gray: np.ndarray, erode_frac: float = 0.04) -> np.ndarray:
    """Endoscope field of view: the largest bright region, eroded so that its own border and the
    burnt-in text next to it (date, settings) contribute no edges."""
    fov = (gray > 20).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fov, connectivity=8)
    if n <= 1:
        return np.ones(gray.shape, dtype=bool)
    fov = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    k = max(3, int(erode_frac * gray.shape[1]) | 1)
    return cv2.erode(fov, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0


def laplacian_variance(rgb: np.ndarray, width: int = 256) -> float:
    """Variance of the Laplacian over mucosa only: low values mean a blurry frame.

    The frame is resized to a fixed width so the score does not depend on resolution. Pixels
    outside the field of view, on device overlays (ScopeGuide box, text) and around specular
    highlights are excluded: they are sharp in every frame, blurry or not, and otherwise
    dominate the score.
    """
    from .overlays import device_overlay_mask

    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    exclude = cv2.dilate((specular_mask(small) | device_overlay_mask(small)).astype(np.uint8), k)
    valid = field_of_view(gray) & (exclude == 0)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap[valid].var()) if valid.sum() > 0.05 * valid.size else float(lap.var())


def specular_mask(rgb: np.ndarray, v_min: int = 230, s_max: int = 40) -> np.ndarray:
    """Specular highlights: very bright, almost unsaturated pixels."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return (hsv[..., 2] >= v_min) & (hsv[..., 1] <= s_max)


def specular_ratio(rgb: np.ndarray, region: np.ndarray | None = None) -> float:
    spec = specular_mask(rgb)
    if region is not None:
        region = region.astype(bool)
        return float(spec[region].mean()) if region.any() else 0.0
    return float(spec.mean())


def mean_brightness(rgb: np.ndarray) -> float:
    return float(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 2].mean())


@dataclass
class QualityReport:
    blur: float
    specular: float
    brightness: float
    reasons: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.reasons


@dataclass
class QualityFilter:
    """Thresholds are calibrated on the data (see ``python -m polypseg.quality --calibrate``)."""

    min_blur: float = 30.0
    max_specular: float = 0.15
    min_brightness: float = 40.0
    max_brightness: float = 235.0

    def assess(self, rgb: np.ndarray) -> QualityReport:
        r = QualityReport(laplacian_variance(rgb), specular_ratio(rgb), mean_brightness(rgb))
        if r.blur < self.min_blur:
            r.reasons.append("blur")
        if r.specular > self.max_specular:
            r.reasons.append("specular")
        if r.brightness < self.min_brightness:
            r.reasons.append("dark")
        if r.brightness > self.max_brightness:
            r.reasons.append("overexposed")
        return r


def _calibrate() -> None:
    import argparse

    from .data import cvc_samples, kvasir_samples
    from .preprocess import read_image

    ap = argparse.ArgumentParser(description="Print quality-score distributions to pick thresholds.")
    ap.add_argument("--kvasir", default="data/raw/Kvasir-SEG")
    ap.add_argument("--cvc", default="data/raw/cvc/CVC-ClinicDB")
    args = ap.parse_args()
    qs = [1, 5, 10, 25, 50]
    for name, samples in [("kvasir", kvasir_samples(args.kvasir)), ("cvc", cvc_samples(args.cvc))]:
        blur, spec, bright = [], [], []
        for s in samples:
            rgb = read_image(s.image_path)
            blur.append(laplacian_variance(rgb))
            spec.append(specular_ratio(rgb))
            bright.append(mean_brightness(rgb))
        print(f"{name:7s} blur  p{qs}: {np.percentile(blur, qs).round(1).tolist()}")
        print(f"{name:7s} spec  p[50,90,99]: {np.percentile(spec, [50, 90, 99]).round(3).tolist()}")
        print(f"{name:7s} bright p[1,50,99]: {np.percentile(bright, [1, 50, 99]).round(1).tolist()}")


if __name__ == "__main__":
    _calibrate()
