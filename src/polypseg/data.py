"""Dataset indexing and leakage-safe splitting for Kvasir-SEG and CVC-ClinicDB."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess import normalize, read_image, read_mask

# Frame-number ranges of the 29 video sequences, copied from the README shipped with CVC-ClinicDB.
# Frames from one sequence show the same polyp from slightly different viewpoints, so they must
# never be split between train and test.
CVC_SEQUENCE_RANGES: tuple[tuple[int, int], ...] = (
    (1, 25), (26, 50), (51, 67), (68, 78), (79, 103), (104, 126), (127, 151), (152, 177),
    (178, 199), (200, 205), (206, 227), (228, 252), (253, 277), (278, 297), (298, 317),
    (318, 342), (343, 363), (364, 383), (384, 408), (409, 428), (429, 447), (448, 466),
    (467, 478), (479, 503), (504, 528), (529, 546), (547, 571), (572, 591), (592, 612),
)  # fmt: skip


def cvc_sequence_id(frame: int) -> int:
    """Map a CVC-ClinicDB frame number (1..612) to its sequence id (1..29)."""
    for seq, (lo, hi) in enumerate(CVC_SEQUENCE_RANGES, start=1):
        if lo <= frame <= hi:
            return seq
    raise ValueError(f"CVC-ClinicDB frame {frame} is outside 1..612")


@dataclass(frozen=True)
class Sample:
    name: str
    image_path: Path
    mask_path: Path | None  # None for a polyp-free (negative) frame
    group: str  # leakage unit: video sequence for CVC, the image itself for Kvasir


def kvasir_samples(root: str | Path) -> list[Sample]:
    root = Path(root)
    samples = []
    for img in sorted((root / "images").glob("*.jpg")):
        mask = root / "masks" / img.name
        if not mask.exists():
            raise FileNotFoundError(f"missing mask for {img.name}")
        # Kvasir-SEG ships no patient or video ids, so each image is its own group.
        samples.append(Sample(img.stem, img, mask, group=img.stem))
    if not samples:
        raise FileNotFoundError(f"no Kvasir-SEG images under {root}")
    return samples


def cvc_samples(root: str | Path) -> list[Sample]:
    root = Path(root)
    images = sorted((root / "Original").glob("*.tif"), key=lambda p: int(p.stem))
    samples = []
    for img in images:
        frame = int(img.stem)
        mask = root / "Ground Truth" / img.name
        if not mask.exists():
            raise FileNotFoundError(f"missing mask for {img.name}")
        samples.append(Sample(f"cvc_{frame:03d}", img, mask, group=f"seq{cvc_sequence_id(frame):02d}"))
    if not samples:
        raise FileNotFoundError(f"no CVC-ClinicDB images under {root}")
    return samples


def negative_samples(root: str | Path) -> list[Sample]:
    """Polyp-free frames (empty masks), one sub-folder per HyperKvasir class.

    Kvasir-SEG has a polyp in every image, so a model trained on it alone never learns that a
    fold or a close-up of normal mucosa is *not* a polyp.
    """
    root = Path(root)
    images = sorted(root.glob("*/*.jpg"))
    if not images:
        raise FileNotFoundError(f"no negative images under {root}")
    return [Sample(f"neg_{p.parent.name}_{p.stem}", p, None, group=p.stem) for p in images]


def group_split(
    samples: Sequence[Sample], val_fraction: float, seed: int
) -> tuple[list[Sample], list[Sample]]:
    """Split by `Sample.group` so no group lands on both sides.

    Groups are shuffled and assigned to validation until it holds at least `val_fraction`
    of the samples. With one sample per group this is an ordinary random split.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    groups = sorted({s.group for s in samples})
    random.Random(seed).shuffle(groups)
    target = val_fraction * len(samples)
    by_group: dict[str, list[Sample]] = {}
    for s in samples:
        by_group.setdefault(s.group, []).append(s)
    val_groups: set[str] = set()
    n_val = 0
    for g in groups:
        if n_val >= target:
            break
        val_groups.add(g)
        n_val += len(by_group[g])
    train = [s for s in samples if s.group not in val_groups]
    val = [s for s in samples if s.group in val_groups]
    return train, val


class SegmentationDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], transform: Callable | None = None):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        image = read_image(s.image_path)
        mask = (
            read_mask(s.mask_path)
            if s.mask_path is not None
            else np.zeros(image.shape[:2], dtype=np.uint8)
        )
        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
        x = torch.from_numpy(normalize(image))
        y = torch.from_numpy(mask.astype(np.float32))[None]
        return x, y
