"""Latency benchmark across runtimes at batch size 1 (the real-time case).

    python -m polypseg.benchmark --ckpt runs/unet_r34/best.pt --onnx runs/unet_r34/best.onnx

Backends are skipped when unavailable: PyTorch CUDA/MPS/CPU, ONNX Runtime (CUDA / CoreML / CPU)
and TensorRT FP16 (built from the ONNX file with the ``tensorrt`` Python package, NVIDIA GPU only).
Only the network forward pass is timed; pre/post-processing is identical for all backends.
"""

from __future__ import annotations

import argparse
import platform
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .model import load_checkpoint


def time_fn(fn: Callable[[], None], warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    return {"mean_ms": ts.mean(), "p50_ms": np.median(ts), "p95_ms": np.percentile(ts, 95),
            "fps": 1000 / ts.mean()}  # fmt: skip


def torch_backend(model, device: str, size: int, half: bool = False) -> Callable[[], None]:
    dev = torch.device(device)
    m = model.to(dev).eval()
    x = torch.randn(1, 3, size, size, device=dev)
    if half:
        m, x = m.half(), x.half()
    sync = {"cuda": torch.cuda.synchronize, "mps": torch.mps.synchronize}.get(device, lambda: None)

    @torch.no_grad()
    def run() -> None:
        m(x)
        sync()

    return run


def ort_backend(path: Path, provider: str, size: int) -> Callable[[], None]:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=[provider])
    if sess.get_providers()[0] != provider:
        raise RuntimeError(f"{provider} not active")
    x = np.random.default_rng(0).standard_normal((1, 3, size, size), dtype=np.float32)
    return lambda: sess.run(None, {"image": x})


def tensorrt_backend(path: Path, size: int) -> Callable[[], None]:
    """Build an FP16 engine from the ONNX file and time execution (requires tensorrt + CUDA)."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(path.read_bytes()):
        raise RuntimeError(parser.get_error(0))
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    shape = (1, 3, size, size)
    profile.set_shape("image", shape, shape, shape)
    config.add_optimization_profile(profile)
    engine = trt.Runtime(logger).deserialize_cuda_engine(
        builder.build_serialized_network(network, config)
    )
    ctx = engine.create_execution_context()
    ctx.set_input_shape("image", shape)
    x = torch.randn(*shape, device="cuda")
    y = torch.empty(tuple(ctx.get_tensor_shape("logits")), device="cuda")
    ctx.set_tensor_address("image", x.data_ptr())
    ctx.set_tensor_address("logits", y.data_ptr())
    stream = torch.cuda.current_stream().cuda_stream

    def run() -> None:
        ctx.execute_async_v3(stream)
        torch.cuda.synchronize()

    return run


def hardware_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    if platform.system() == "Darwin":
        import subprocess

        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True).stdout.strip()  # fmt: skip
    return platform.processor() or platform.machine()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--onnx", default=None, help="default: checkpoint path with .onnx suffix")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    ap.add_argument("--out", default=None, help="CSV path")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    model, cfg, _ = load_checkpoint(args.ckpt, "cpu")
    size = cfg["data"]["image_size"]
    onnx_path = Path(args.onnx or Path(args.ckpt).with_suffix(".onnx"))

    candidates: list[tuple[str, Callable[[], Callable[[], None]]]] = [
        ("PyTorch CPU FP32", lambda: torch_backend(model, "cpu", size)),
        ("ONNX Runtime CPU FP32", lambda: ort_backend(onnx_path, "CPUExecutionProvider", size)),
    ]
    if torch.backends.mps.is_available():
        candidates.append(("PyTorch MPS FP32", lambda: torch_backend(model, "mps", size)))
        candidates.append(
            ("ONNX Runtime CoreML", lambda: ort_backend(onnx_path, "CoreMLExecutionProvider", size))
        )
    if torch.cuda.is_available():
        candidates += [
            ("PyTorch CUDA FP32", lambda: torch_backend(model, "cuda", size)),
            ("PyTorch CUDA FP16", lambda: torch_backend(model, "cuda", size, half=True)),
            ("ONNX Runtime CUDA FP32",
             lambda: ort_backend(onnx_path, "CUDAExecutionProvider", size)),
            ("TensorRT FP16", lambda: tensorrt_backend(onnx_path, size)),
        ]  # fmt: skip

    rows = []
    for name, make in candidates:
        try:
            r = time_fn(make(), args.warmup, args.iters)
        except Exception as e:  # report and continue: a missing backend is not fatal
            print(f"skip {name}: {type(e).__name__}: {e}")
            continue
        rows.append({"backend": name, **r})
        print(f"{name:26s} {r['mean_ms']:7.2f} ms  p95 {r['p95_ms']:7.2f} ms  {r['fps']:6.1f} FPS")
    df = pd.DataFrame(rows).round(2)
    df.insert(0, "hardware", hardware_name())
    df.insert(1, "input", f"1x3x{size}x{size}")
    out = Path(args.out or f"outputs/{cfg['run_name']}/benchmark.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)


if __name__ == "__main__":
    main()
