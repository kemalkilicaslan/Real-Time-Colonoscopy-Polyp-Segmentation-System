"""On-screen device overlays that must never be reported as polyps.

Olympus ScopeGuide draws a saturated teal box (scope position map) in the corner of the video.
Only ~1% of Kvasir-SEG training images contain it, so the network has never learned that it is
background and fires on it with high confidence. The box is a synthetic, perfectly uniform
colour that does not occur in mucosa, so it is masked out deterministically.
"""

from __future__ import annotations

import cv2
import numpy as np

# OpenCV HSV (H in 0..180). Mucosa is red/pink/orange (H < 25 or > 160); bile is yellow-green.
TEAL_LO = (75, 120, 80)
TEAL_HI = (100, 255, 255)


def device_overlay_mask(
    rgb: np.ndarray, min_area_frac: float = 0.005, min_fill: float = 0.6, pad: int = 6
) -> np.ndarray:
    """Boolean mask of rectangular teal overlay boxes (padded), all False if none is present."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    teal = cv2.inRange(hsv, TEAL_LO, TEAL_HI)
    out = np.zeros(teal.shape, dtype=bool)
    n, _, stats, _ = cv2.connectedComponentsWithStats(teal, connectivity=8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_area_frac * teal.size and area >= min_fill * w * h:
            out[max(0, y - pad) : y + h + pad, max(0, x - pad) : x + w + pad] = True
    return out


def ignore_mask(rgb: np.ndarray) -> np.ndarray:
    """Pixels where a detection can never be a polyp: device overlays and everything outside the
    endoscope's field of view (black frame with burnt-in patient/date/settings text)."""
    from .quality import field_of_view

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return device_overlay_mask(rgb) | ~field_of_view(gray, erode_frac=0.01)
