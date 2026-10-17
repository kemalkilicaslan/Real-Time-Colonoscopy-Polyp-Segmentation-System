from pathlib import Path

import pytest

from polypseg.data import CVC_SEQUENCE_RANGES, Sample, cvc_sequence_id, group_split


def test_cvc_ranges_cover_all_612_frames_without_gaps():
    covered = [f for lo, hi in CVC_SEQUENCE_RANGES for f in range(lo, hi + 1)]
    assert covered == list(range(1, 613))
    assert len(CVC_SEQUENCE_RANGES) == 29


@pytest.mark.parametrize("frame,seq", [(1, 1), (25, 1), (26, 2), (205, 10), (612, 29)])
def test_cvc_sequence_id(frame, seq):
    assert cvc_sequence_id(frame) == seq


def test_cvc_sequence_id_out_of_range():
    with pytest.raises(ValueError):
        cvc_sequence_id(613)


def fake_cvc():
    return [
        Sample(f"cvc_{f}", Path("x"), Path("y"), f"seq{cvc_sequence_id(f):02d}")
        for f in range(1, 613)
    ]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_group_split_never_shares_a_sequence(seed):
    train, val = group_split(fake_cvc(), 0.2, seed)
    assert {s.group for s in train}.isdisjoint({s.group for s in val})
    assert len(train) + len(val) == 612
    assert len(val) >= 0.2 * 612


def test_group_split_is_deterministic():
    a = group_split(fake_cvc(), 0.2, 7)[1]
    b = group_split(fake_cvc(), 0.2, 7)[1]
    assert [s.name for s in a] == [s.name for s in b]


def test_group_split_rejects_bad_fraction():
    with pytest.raises(ValueError):
        group_split(fake_cvc(), 1.0, 0)


def test_negative_samples_yield_empty_masks(tmp_path):
    import cv2
    import numpy as np

    from polypseg.data import SegmentationDataset, negative_samples

    (tmp_path / "cecum").mkdir()
    cv2.imwrite(str(tmp_path / "cecum" / "a.jpg"), np.full((40, 50, 3), 120, np.uint8))
    samples = negative_samples(tmp_path)
    assert samples[0].mask_path is None and samples[0].name == "neg_cecum_a"
    x, y = SegmentationDataset(samples)[0]
    assert x.shape == (3, 40, 50) and y.shape == (1, 40, 50) and y.sum() == 0
