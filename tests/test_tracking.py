import numpy as np
import pytest

from polypseg.tracking import IoUTracker, box_iou, mask_to_boxes


def test_box_iou():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_mask_to_boxes_exclusive_coords_and_min_area():
    m = np.zeros((50, 50), np.uint8)
    m[10:20, 5:15] = 1
    m[40, 40] = 1
    assert mask_to_boxes(m, min_area=2) == [(5, 10, 15, 20)]


def test_alarm_only_after_min_hits_consecutive_frames():
    tr = IoUTracker(min_hits=3)
    box = (10, 10, 50, 50)
    assert tr.update([box], 0) == []
    assert tr.update([box], 1) == []
    confirmed = tr.update([box], 2)
    assert len(confirmed) == 1 and confirmed[0].confirmed_frame == 2
    assert tr.update([box], 3) == []  # no re-alarm for the same polyp


def test_single_frame_flicker_never_alarms():
    tr = IoUTracker(min_hits=3)
    for i in range(10):
        boxes = [(10, 10, 50, 50)] if i % 2 == 0 else []
        assert tr.update(boxes, i) == []
    assert not any(t.confirmed for t in tr.finalize())


def test_confirmed_track_survives_short_gap_and_keeps_id():
    tr = IoUTracker(min_hits=2, max_misses=2)
    box = (10, 10, 50, 50)
    tr.update([box], 0)
    (t,) = tr.update([box], 1)
    tr.update([], 2)
    tr.update([], 3)
    assert tr.update([(12, 12, 52, 52)], 4) == []
    assert tr.confirmed_tracks()[0].track_id == t.track_id


def test_track_dies_after_max_misses():
    tr = IoUTracker(min_hits=1, max_misses=1)
    tr.update([(0, 0, 10, 10)], 0)
    tr.update([], 1)
    tr.update([], 2)
    assert tr.active == []
    assert len(tr.update([(0, 0, 10, 10)], 3)) == 1  # new track, new alarm


def test_two_polyps_get_two_tracks():
    tr = IoUTracker(min_hits=1)
    new = tr.update([(0, 0, 10, 10), (50, 50, 70, 70)], 0)
    assert len({t.track_id for t in new}) == 2
