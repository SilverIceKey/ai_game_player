"""dataset/episode.py 单元测试：手动 START/STOP 状态机边界行为。"""
from __future__ import annotations

import pytest

from capture.action import SOURCE_CORRECTION, SOURCE_HUMAN
from dataset.episode import EpisodeTracker


def test_start_stop_flow():
    tracker = EpisodeTracker()
    assert not tracker.is_active
    assert tracker.current_episode_id is None

    assert tracker.start(1_000_000) is True
    assert tracker.is_active
    assert tracker.current_episode_id == 0

    meta = tracker.stop(2_000_000)
    assert meta is not None
    assert meta.episode_id == 0
    assert meta.start_us == 1_000_000
    assert meta.end_us == 2_000_000
    assert meta.source == SOURCE_HUMAN
    assert not tracker.is_active
    assert tracker.current_episode_id is None


def test_episode_id_increments():
    tracker = EpisodeTracker()
    tracker.start(100)
    meta0 = tracker.stop(200)
    tracker.start(300, source=SOURCE_CORRECTION)
    meta1 = tracker.stop(400)
    assert meta0 is not None and meta0.episode_id == 0
    assert meta1 is not None and meta1.episode_id == 1
    assert meta1.source == SOURCE_CORRECTION


def test_stop_without_start_ignored():
    tracker = EpisodeTracker()
    assert tracker.stop(1_000_000) is None  # 忽略，不抛异常
    assert not tracker.is_active


def test_double_start_ignored():
    tracker = EpisodeTracker()
    assert tracker.start(1_000_000) is True
    assert tracker.start(1_500_000) is False  # 重复 start 被忽略，原 episode 继续
    meta = tracker.stop(2_000_000)
    assert meta is not None
    assert meta.start_us == 1_000_000  # 起始时间保持第一次 start 的值


def test_stop_before_start_raises():
    tracker = EpisodeTracker()
    tracker.start(2_000_000)
    with pytest.raises(ValueError, match="时间倒流"):
        tracker.stop(1_000_000)
