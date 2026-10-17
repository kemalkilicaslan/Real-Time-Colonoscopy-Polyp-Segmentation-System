"""Model construction, device selection and checkpoint I/O."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import segmentation_models_pytorch as smp
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(arch: str, encoder: str, encoder_weights: str | None = None) -> torch.nn.Module:
    """Binary segmentation network that outputs raw logits of shape (N, 1, H, W)."""
    return smp.create_model(
        arch=arch, encoder_name=encoder, encoder_weights=encoder_weights, in_channels=3, classes=1
    )


def pick_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(path: str | Path, model: torch.nn.Module, cfg: dict, **extra: Any) -> None:
    torch.save({"state_dict": model.state_dict(), "config": cfg, **extra}, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    """Returns (model in eval mode, config dict, checkpoint dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    m = ckpt["config"]["model"]
    model = build_model(m["arch"], m["encoder"], encoder_weights=None)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["config"], ckpt
