"""Real-time video pipeline: FFmpeg decode -> quality gate -> segmentation -> IoU tracking -> alarm.

    python -m polypseg.video --model runs/unet_r34/best.onnx --input colonoscopy.mp4 \
        --output outputs/demo.mp4
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .inference import load_segmenter
from .metrics import false_alarms_per_minute, remove_small
from .overlays import ignore_mask
from .quality import QualityFilter, QualityReport
from .tracking import IoUTracker, Track, mask_to_boxes


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    n_frames: int | None


def probe(path: str | Path) -> VideoInfo:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
           "-show_entries", "stream=width,height,avg_frame_rate,nb_read_packets",
           "-of", "json", str(path)]  # fmt: skip
    s = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)
    s = s["streams"][0]
    num, den = (int(v) for v in s["avg_frame_rate"].split("/"))
    n = s.get("nb_read_packets")
    return VideoInfo(int(s["width"]), int(s["height"]), num / den if den else 25.0,
                     int(n) if n else None)  # fmt: skip


def read_frames(path: str | Path, info: VideoInfo | None = None) -> Iterator[np.ndarray]:
    """Decode with FFmpeg into raw RGB24 frames over a pipe (no OpenCV backend quirks)."""
    info = info or probe(path)
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    frame_bytes = info.width * info.height * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield np.frombuffer(buf, np.uint8).reshape(info.height, info.width, 3)
    finally:
        if proc.poll() is None:  # consumer stopped early: stop ffmpeg before closing its pipe
            proc.kill()
        proc.stdout.close()
        proc.wait()


class FrameWriter:
    """Encode RGB frames to H.264 through an FFmpeg pipe."""

    def __init__(self, path: str | Path, width: int, height: int, fps: float):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(path)]  # fmt: skip
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, rgb: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(rgb).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait()


@dataclass
class FrameResult:
    index: int
    quality: QualityReport | None
    mask: np.ndarray | None  # None when the frame was skipped by the quality gate
    new_alarms: list[Track] = field(default_factory=list)
    shown: list[Track] = field(default_factory=list)
    infer_ms: float = 0.0

    @property
    def skipped(self) -> bool:
        return self.mask is None


class VideoPipeline:
    def __init__(
        self,
        segmenter,
        threshold: float = 0.5,
        min_area_frac: float = 0.001,
        quality: QualityFilter | None = None,
        tracker: IoUTracker | None = None,
        mask_overlays: bool = True,
    ):
        self.segmenter = segmenter
        self.mask_overlays = mask_overlays
        self.threshold = threshold
        self.min_area_frac = min_area_frac
        self.quality = quality
        self.tracker = tracker or IoUTracker()

    def process(self, rgb: np.ndarray, index: int) -> FrameResult:
        q = self.quality.assess(rgb) if self.quality else None
        if q is not None and not q.ok:
            # Unusable frame: do not infer and do not count it as a miss, so a confirmed polyp
            # survives a few blurry frames without re-triggering the alarm afterwards.
            return FrameResult(index, q, None, shown=self.tracker.confirmed_tracks())
        t0 = time.perf_counter()
        prob = self.segmenter(rgb)
        infer_ms = (time.perf_counter() - t0) * 1000
        if self.mask_overlays:
            prob[ignore_mask(rgb)] = 0.0
        min_area = max(1, int(self.min_area_frac * prob.size))
        mask = remove_small(prob > self.threshold, min_area)
        new = self.tracker.update(mask_to_boxes(mask, min_area), index)
        return FrameResult(index, q, mask, new, self.tracker.confirmed_tracks(), infer_ms)


GREEN, AMBER, RED = (40, 220, 90), (255, 190, 0), (255, 60, 60)


def draw(rgb: np.ndarray, r: FrameResult, fps_text: str = "") -> np.ndarray:
    out = rgb.copy()
    shown_ids = {t.track_id for t in r.shown}
    if r.mask is not None and shown_ids:
        # Only overlay pixels of confirmed tracks; tentative blobs stay hidden (debounced).
        overlay = out.copy()
        for t in r.shown:
            x1, y1, x2, y2 = t.box
            region = r.mask[y1:y2, x1:x2].astype(bool)
            overlay[y1:y2, x1:x2][region] = GREEN
        out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
    for t in r.shown:
        x1, y1, x2, y2 = t.box
        cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), GREEN, 2)
        cv2.putText(out, f"polyp #{t.track_id}", (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1, cv2.LINE_AA)  # fmt: skip
    status = "SKIPPED: " + ",".join(r.quality.reasons) if r.skipped and r.quality else fps_text
    color = AMBER if r.skipped else (220, 220, 220)
    cv2.putText(out, status, (10, out.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                cv2.LINE_AA)  # fmt: skip
    if r.new_alarms:
        cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), RED, 6)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", required=True, help=".pt checkpoint or .onnx model")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=None, help="annotated mp4 (optional)")
    ap.add_argument("--events", default=None, help="CSV of alarm events (optional)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--providers", nargs="*", default=None, help="ONNX Runtime providers")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-hits", type=int, default=3, help="consecutive frames before an alarm")
    ap.add_argument("--max-misses", type=int, default=5)
    ap.add_argument("--min-blur", type=float, default=QualityFilter.min_blur)
    ap.add_argument("--no-quality", action="store_true", help="disable the quality gate")
    ap.add_argument("--no-overlay-mask", action="store_true", help="keep overlays and area outside FOV")
    ap.add_argument("--polyp-free", action="store_true",
                    help="video is known to contain no polyp: every alarm is a false alarm")  # fmt: skip
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found on PATH")
    info = probe(args.input)
    pipe = VideoPipeline(
        load_segmenter(args.model, args.device, args.providers),
        threshold=args.threshold,
        quality=None if args.no_quality else QualityFilter(min_blur=args.min_blur),
        tracker=IoUTracker(min_hits=args.min_hits, max_misses=args.max_misses),
        mask_overlays=not args.no_overlay_mask,
    )
    writer = FrameWriter(args.output, info.width, info.height, info.fps) if args.output else None
    n, skipped, infer_ms, t_start = 0, 0, [], time.perf_counter()
    for i, rgb in enumerate(read_frames(args.input, info)):
        r = pipe.process(rgb, i)
        n += 1
        skipped += r.skipped
        if not r.skipped:
            infer_ms.append(r.infer_ms)
        if writer:
            fps_now = 1000 / np.mean(infer_ms[-30:]) if infer_ms else 0.0
            writer.write(draw(rgb, r, f"{fps_now:.0f} FPS (model)"))
    if writer:
        writer.close()
    wall = time.perf_counter() - t_start

    tracks = pipe.tracker.finalize()
    alarms = [t for t in tracks if t.confirmed]
    if args.events:
        with open(args.events, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["track_id", "first_frame", "alarm_frame", "last_frame", "alarm_time_s"])
            for t in alarms:
                w.writerow([t.track_id, t.first_frame, t.confirmed_frame, t.history[-1][0],
                            f"{t.confirmed_frame / info.fps:.2f}"])  # fmt: skip
    stats = {
        "frames": n,
        "duration_s": round(n / info.fps, 1),
        "skipped_by_quality_gate": skipped,
        "skipped_pct": round(100 * skipped / max(1, n), 1),
        "alarms": len(alarms),
        "suppressed_tentative_tracks": len(tracks) - len(alarms),
        "model_ms_mean": round(float(np.mean(infer_ms)), 2) if infer_ms else None,
        "end_to_end_fps": round(n / wall, 1),
    }
    if args.polyp_free:
        stats["false_alarms_per_min"] = round(false_alarms_per_minute(len(alarms), n, info.fps), 2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
