"""IoU tracker that turns per-frame blobs into stable, debounced alarms.

A single-frame detection is not shown to the endoscopist: a track has to be matched in
``min_hits`` consecutive frames before it is *confirmed* and raises an alarm. This removes most
one-frame flicker (bubbles, highlights) at the cost of ``min_hits - 1`` frames of latency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

Box = tuple[int, int, int, int]  # x1, y1, x2, y2 (exclusive)


def mask_to_boxes(mask: np.ndarray, min_area: int = 1) -> list[Box]:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            boxes.append((int(x), int(y), int(x + w), int(y + h)))
    return boxes


def box_iou(a: Box, b: Box) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    box: Box
    first_frame: int
    hits: int = 1
    misses: int = 0
    confirmed_frame: int | None = None
    history: list[tuple[int, Box]] = field(default_factory=list)

    @property
    def confirmed(self) -> bool:
        return self.confirmed_frame is not None


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.2, min_hits: int = 3, max_misses: int = 5):
        if min_hits < 1:
            raise ValueError("min_hits must be >= 1")
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits
        self.max_misses = max_misses
        self.active: list[Track] = []
        self.finished: list[Track] = []
        self._next_id = 1

    def update(self, boxes: list[Box], frame_idx: int) -> list[Track]:
        """Associate this frame's boxes with active tracks. Returns tracks confirmed this frame."""
        pairs = sorted(
            (
                (box_iou(t.box, b), ti, bi)
                for ti, t in enumerate(self.active)
                for bi, b in enumerate(boxes)
            ),
            reverse=True,
        )
        used_t, used_b = set(), set()
        for score, ti, bi in pairs:  # greedy, highest IoU first
            if score < self.iou_threshold:
                break
            if ti in used_t or bi in used_b:
                continue
            used_t.add(ti)
            used_b.add(bi)
            t = self.active[ti]
            t.box, t.misses = boxes[bi], 0
            t.hits += 1
            t.history.append((frame_idx, t.box))

        newly_confirmed, survivors = [], []
        for ti, t in enumerate(self.active):
            if ti not in used_t:
                t.misses += 1
                # A tentative track must be seen in consecutive frames; a confirmed one may
                # survive short occlusions (blur, scope motion).
                if not t.confirmed or t.misses > self.max_misses:
                    self.finished.append(t)
                    continue
            if not t.confirmed and t.hits >= self.min_hits:
                t.confirmed_frame = frame_idx
                newly_confirmed.append(t)
            survivors.append(t)

        for bi, b in enumerate(boxes):
            if bi not in used_b:
                t = Track(self._next_id, b, first_frame=frame_idx, history=[(frame_idx, b)])
                self._next_id += 1
                if self.min_hits == 1:
                    t.confirmed_frame = frame_idx
                    newly_confirmed.append(t)
                survivors.append(t)
        self.active = survivors
        return newly_confirmed

    def confirmed_tracks(self) -> list[Track]:
        return [t for t in self.active if t.confirmed and t.misses == 0]

    def finalize(self) -> list[Track]:
        """End of video: close all tracks and return every track ever created."""
        self.finished.extend(self.active)
        self.active = []
        return self.finished
