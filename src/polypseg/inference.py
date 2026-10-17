"""Backend-agnostic single-frame inference: RGB frame in, probability map (original size) out."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .model import load_checkpoint, pick_device
from .preprocess import postprocess, preprocess


class TorchSegmenter:
    def __init__(self, model: torch.nn.Module, image_size: int, device: torch.device):
        self.model, self.image_size, self.device = model.eval(), image_size, device

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(preprocess(rgb, self.image_size)).to(self.device)
        return postprocess(self.model(x).float().cpu().numpy(), rgb.shape[:2])


class OnnxSegmenter:
    def __init__(self, path: str | Path, providers: list[str] | None = None):
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            str(path), providers=providers or ["CPUExecutionProvider"]
        )
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.image_size = int(inp.shape[-1])

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        x = preprocess(rgb, self.image_size)
        (logits,) = self.session.run(None, {self.input_name: x})
        return postprocess(logits, rgb.shape[:2])


def load_segmenter(path: str | Path, device: str | None = None, providers: list[str] | None = None):
    """``.pt`` checkpoint -> PyTorch backend, ``.onnx`` -> ONNX Runtime backend."""
    path = Path(path)
    if path.suffix == ".onnx":
        return OnnxSegmenter(path, providers)
    dev = pick_device(device)
    model, cfg, _ = load_checkpoint(path, dev)
    return TorchSegmenter(model, cfg["data"]["image_size"], dev)
