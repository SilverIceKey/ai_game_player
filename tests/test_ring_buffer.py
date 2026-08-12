"""runtime/ring_buffer.py 单元测试（spec §30 History Ring Buffer）。"""
from __future__ import annotations

import threading

import numpy as np
import pytest

from capture.action import ActionRecord, NormalizedAction, SOURCE_AI, SOURCE_HUMAN
from runtime.ring_buffer import ActionHistoryBuffer, FrameRingBuffer


def _frame(v: int) -> np.ndarray:
    return np.full((2, 2, 3), v, dtype=np.uint8)


class TestFrameRingBuffer:
    def test_capacity_must_be_positive(self):
        with pytest.raises(ValueError):
            FrameRingBuffer(0)

    def test_latest_empty_returns_none(self):
        assert FrameRingBuffer(4).latest() is None

    def test_push_and_latest(self):
        buf = FrameRingBuffer(4)
        buf.push(_frame(1), 1000)
        buf.push(_frame(2), 2000)
        frame, ts = buf.latest()
        assert ts == 2000
        assert frame[0, 0, 0] == 2

    def test_window_chronological_order(self):
        buf = FrameRingBuffer(8)
        for i in range(5):
            buf.push(_frame(i), i * 1000)
        window = buf.window(3)
        assert [ts for _, ts in window] == [2000, 3000, 4000]

    def test_window_short_returns_all(self):
        buf = FrameRingBuffer(8)
        buf.push(_frame(1), 1000)
        assert len(buf.window(4)) == 1

    def test_window_zero_returns_empty(self):
        buf = FrameRingBuffer(8)
        buf.push(_frame(1), 1000)
        assert buf.window(0) == []

    def test_capacity_evicts_oldest(self):
        buf = FrameRingBuffer(3)
        for i in range(5):
            buf.push(_frame(i), i * 1000)
        assert len(buf) == 3
        assert [ts for _, ts in buf.window(3)] == [2000, 3000, 4000]

    def test_concurrent_push_and_window(self):
        """Capture 线程推帧与推理线程取窗并发：不抛异常、长度不超容量。"""
        buf = FrameRingBuffer(16)
        errors: list[Exception] = []

        def producer():
            try:
                for i in range(500):
                    buf.push(_frame(i % 255), i * 1000)
            except Exception as exc:
                errors.append(exc)

        def consumer():
            try:
                for _ in range(500):
                    buf.window(4)
                    buf.latest()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=producer) for _ in range(2)] + [
            threading.Thread(target=consumer) for _ in range(2)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert len(buf) <= 16


class TestActionHistoryBuffer:
    def test_recent_returns_normalized_actions(self):
        buf = ActionHistoryBuffer(8)
        buf.push(ActionRecord(1000, NormalizedAction(move_y=1.0), SOURCE_HUMAN))
        buf.push(ActionRecord(2000, NormalizedAction(buttons=frozenset({"dodge"})), SOURCE_AI))
        recent = buf.recent(2)
        assert [a.move_y for a in recent] == [1.0, 0.0]
        assert recent[1].pressed("dodge")

    def test_recent_short_returns_all(self):
        buf = ActionHistoryBuffer(8)
        buf.push(ActionRecord(1000, NormalizedAction.neutral(), SOURCE_HUMAN))
        assert len(buf.recent(4)) == 1

    def test_capacity_evicts_oldest(self):
        buf = ActionHistoryBuffer(2)
        for i in range(4):
            buf.push(ActionRecord(i * 1000, NormalizedAction(move_x=float(i) / 10), SOURCE_AI))
        actions = buf.recent(2)
        assert [a.move_x for a in actions] == [0.2, 0.3]
