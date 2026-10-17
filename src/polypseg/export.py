"""Export a checkpoint to ONNX and check it numerically against PyTorch.

    python -m polypseg.export --ckpt runs/unet_r34/best.pt            # -> runs/unet_r34/best.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .model import load_checkpoint


def export_onnx(model: torch.nn.Module, path: str | Path, image_size: int, opset: int = 17) -> Path:
    """Fixed spatial size (what TensorRT optimises best), dynamic batch."""
    model = model.cpu().eval()
    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        dynamo=False,
    )
    return Path(path)


def check_onnx(model: torch.nn.Module, path: str | Path, image_size: int, n: int = 4) -> float:
    """Max absolute logit difference between PyTorch (CPU) and ONNX Runtime on random inputs."""
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(str(path)))
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = torch.randn(n, 3, image_size, image_size)
    with torch.no_grad():
        ref = model.cpu().eval()(x).numpy()
    (got,) = sess.run(None, {"image": x.numpy()})
    return float(np.abs(ref - got).max())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    model, cfg, _ = load_checkpoint(args.ckpt, "cpu")
    size = cfg["data"]["image_size"]
    out = Path(args.out or Path(args.ckpt).with_suffix(".onnx"))
    export_onnx(model, out, size, args.opset)
    err = check_onnx(model, out, size)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB), max |torch - onnx| = {err:.2e}")
    if err > 1e-3:
        raise SystemExit("ONNX output deviates from PyTorch")


if __name__ == "__main__":
    main()
