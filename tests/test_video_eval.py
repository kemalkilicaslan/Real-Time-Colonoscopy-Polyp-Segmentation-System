import pytest

from polypseg.video_eval import replay

POLYP = (10, 10, 50, 50)
ELSEWHERE = (200, 200, 230, 230)


def test_replay_counts_true_and_false_alarms():
    frames = [([POLYP, ELSEWHERE], [POLYP])] * 5
    r = replay(frames, min_hits=3, max_misses=1, iou_thr=0.1)
    assert r["alarms"] == 2 and r["false_alarms"] == 1
    assert r["latency_frames"] == 2  # confirmed on the third frame
    assert r["frames_alarmed"] == 3


@pytest.mark.parametrize("min_hits,expected_alarms", [(1, 3), (2, 0)])
def test_debouncing_suppresses_isolated_false_positives(min_hits, expected_alarms):
    frames = [([ELSEWHERE], [POLYP]), ([], [POLYP])] * 3
    assert replay(frames, min_hits, 0, 0.1)["false_alarms"] == expected_alarms


def test_false_alarm_ablation_replay():
    from polypseg.false_alarms import CachedFrame, Config, replay

    overlay_box, blob = (0, 200, 100, 280), (150, 150, 180, 180)
    frames = [CachedFrame([overlay_box, blob], [blob], low_quality=i % 2 == 1) for i in range(6)]
    # raw: overlay box + blob each become a confirmed track immediately
    assert replay(frames, Config("raw", False, False, 1), 5)[0] == 2
    assert replay(frames, Config("mask", True, False, 1), 5)[0] == 1
    # quality gate drops every other frame; the blob is still seen on 3 usable frames
    assert replay(frames, Config("gate+3", True, True, 3), 5)[0] == 1
    assert replay(frames, Config("gate+5", True, True, 5), 5)[0] == 0
