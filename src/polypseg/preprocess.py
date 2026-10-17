"""Image I/O, normalisation and augmentation shared by training, evaluation and deployment."""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import tifffile

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _read_tiff(path: Path) -> np.ndarray:
    """CVC-ClinicDB stores RGB frames as 3-sample TIFFs tagged photometric=MINISBLACK.

    OpenCV honours the tag and silently returns only the first (red) channel as greyscale, so
    every CVC frame would look grey to the model. tifffile returns the raw samples instead.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    return tifffile.imread(path)


def read_image(path: str | Path) -> np.ndarray:
    """Read an image as RGB uint8 (H, W, 3)."""
    path = Path(path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        img = _read_tiff(path)
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        return np.ascontiguousarray(img[..., :3])
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_mask(path: str | Path, threshold: int = 127) -> np.ndarray:
    """Read a ground-truth mask as uint8 {0, 1}.

    Kvasir-SEG masks are JPEGs and CVC-ClinicDB masks contain anti-aliased edge values,
    so both are binarised at mid-grey rather than tested for == 255.
    """
    path = Path(path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        mask = _read_tiff(path)
        mask = mask[..., 0] if mask.ndim == 3 else mask
    else:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(path)
    return (mask > threshold).astype(np.uint8)


def normalize(rgb: np.ndarray) -> np.ndarray:
    """uint8 RGB (H, W, 3) -> float32 ImageNet-normalised (3, H, W)."""
    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def preprocess(rgb: np.ndarray, size: int) -> np.ndarray:
    """Inference pre-processing: resize to (size, size) and normalise. Returns (1, 3, size, size)."""
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    return normalize(resized)[None]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def postprocess(logits: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Model logits (1, 1, h, w) or (h, w) -> probability map resized to the original frame size.

    Metrics are computed at the original resolution, never at the network resolution.
    """
    logits = np.asarray(logits, dtype=np.float32).squeeze()
    h, w = out_hw
    logits = cv2.resize(logits, (w, h), interpolation=cv2.INTER_LINEAR)
    return sigmoid(logits)


def train_transforms(size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(scale=(0.75, 1.25), translate_percent=(-0.1, 0.1), rotate=(-20, 20), p=0.7),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            A.HueSaturationValue(10, 20, 10, p=0.3),
            # Motion blur is a dominant artefact in colonoscopy video.
            A.OneOf([A.MotionBlur(blur_limit=(3, 9)), A.GaussianBlur(blur_limit=(3, 7))], p=0.25),
        ]
    )


def eval_transforms(size: int) -> A.Compose:
    return A.Compose([A.Resize(size, size)])
