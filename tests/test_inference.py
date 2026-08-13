"""runtime/inference.py 单元测试（spec §30 Inference Worker / §33 日志字段）。

policy 用假实现（固定 ActionChunk），不 import model/ 包，不触达真实推理。
"""
from __future__ import annotations

import threading
import time

import numpy as np

from capture.action import ActionChunk, ActionRecord, NormalizedAction, SOURCE_AI, SOURCE_HUMAN
from config import PredictionConfig
from runtime.inference import InferenceWorker
from runtime.ring_buffer import ActionHistoryBuffer, FrameRingBuffer

HISTORY_FRAMES = 4
HISTORY_ACTIONS = 3


class FakePolicy:
    """鸭子类型假 policy：记录输入，返回固定 chunk。"""

    def __init__(self):
        self.calls: list[tuple[list[np.ndarray], list[NormalizedAction]]] = []

    def predict(self, frames, action_history):
        self.calls.append((frames, action_history))
        return ActionChunk(
            actions=(NormalizedAction(move_y=1.0), NormalizedAction.neutral()),
            step_ms=50.0,
            model_version="model-v001",
            created_us=0,
        )


def _worker(policy: FakePolicy) -> tuple[InferenceWorker, FrameRingBuffer, ActionHistoryBuffer]:
    frames = FrameRingBuffer(8)
    actions = ActionHistoryBuffer(8)
    worker = InferenceWorker(
        policy, frames, actions, HISTORY_FRAMES, HISTORY_ACTIONS,
        prediction=PredictionConfig(action_step_ms=50.0, future_action_steps=4),
    )
    return worker, frames, actions


def _fill_frames(buf: FrameRingBuffer, n: int, base_ts: int = 1_000_000):
    for i in range(n):
        buf.push(np.zeros((2, 2, 3), dtype=np.uint8), base_ts + i * 16_000)


class TestInferOnce:
    def test_window_insufficient_returns_none(self):
        policy = FakePolicy()
        worker, frames, _ = _worker(policy)
        _fill_frames(frames, HISTORY_FRAMES - 1)
        assert worker.infer_once(2_000_000) is None
        assert policy.calls == []  # 窗口不足不触发推理

    def test_normal_path_returns_chunk_and_stats(self):
        policy = FakePolicy()
        worker, frames, actions = _worker(policy)
        _fill_frames(frames, HISTORY_FRAMES)
        actions.push(ActionRecord(500_000, NormalizedAction(move_x=0.5), SOURCE_HUMAN))
        actions.push(ActionRecord(550_000, NormalizedAction.neutral(), SOURCE_AI))

        now = 1_100_000
        result = worker.infer_once(now)
        assert result is not None
        chunk, stats = result
        assert chunk.model_version == "model-v001"
        assert len(chunk.actions) == 2

        # spec §33 必备字段
        assert stats["timestamp_us"] == now
        assert stats["model_version"] == "model-v001"
        # 最新帧 ts = 1_000_000 + 3*16_000 = 1_048_000 → 帧龄 52ms
        assert stats["frame_age_ms"] == (now - 1_048_000) / 1000.0
        assert stats["queue_delay_ms"] >= 0.0
        assert stats["inference_ms"] >= 0.0

    def test_policy_receives_window_and_action_history(self):
        policy = FakePolicy()
        worker, frames, actions = _worker(policy)
        _fill_frames(frames, 6)  # 缓冲 6 帧，窗口只取最近 4 帧
        for i in range(5):
            actions.push(ActionRecord(i * 1000, NormalizedAction(move_x=i / 10), SOURCE_AI))

        worker.infer_once(2_000_000)
        got_frames, got_history = policy.calls[0]
        assert len(got_frames) == HISTORY_FRAMES
        assert all(isinstance(item, tuple) and len(item) == 2 for item in got_frames)  # (frame, ts)
        assert len(got_history) == HISTORY_ACTIONS
        # 动作历史按时间升序、取最近 3 个（ActionRecord 带时间戳，spec §16）
        assert [a.action.move_x for a in got_history] == [0.2, 0.3, 0.4]


class TestRunLoop:
    def test_loop_produces_results_until_stopped(self):
        policy = FakePolicy()
        worker, frames, _ = _worker(policy)
        _fill_frames(frames, HISTORY_FRAMES)

        results: list[tuple[ActionChunk, dict]] = []
        stop = threading.Event()
        thread = threading.Thread(
            target=worker.run_loop,
            kwargs={"stop_event": stop, "on_result": lambda c, s: results.append((c, s)),
                    "interval_s": 0.005},
        )
        thread.start()
        deadline = time.monotonic() + 2.0
        while len(results) < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        stop.set()
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        assert len(results) >= 2
        assert all(c.model_version == "model-v001" for c, _ in results)

    def test_loop_skips_when_window_insufficient(self):
        policy = FakePolicy()
        worker, frames, _ = _worker(policy)
        _fill_frames(frames, 1)  # 窗口不足

        results: list = []
        stop = threading.Event()
        thread = threading.Thread(
            target=worker.run_loop,
            kwargs={"stop_event": stop, "on_result": lambda c, s: results.append(c),
                    "interval_s": 0.005},
        )
        thread.start()
        time.sleep(0.05)
        stop.set()
        thread.join(timeout=2.0)
        assert results == []
        assert policy.calls == []  # 窗口不足不算错误、不推理

    def test_loop_reports_policy_error_via_on_error(self):
        class BoomPolicy:
            def predict(self, frames, action_history):
                raise RuntimeError("boom")

        frames = FrameRingBuffer(8)
        _fill_frames(frames, HISTORY_FRAMES)
        worker = InferenceWorker(BoomPolicy(), frames, ActionHistoryBuffer(8),
                                 HISTORY_FRAMES, HISTORY_ACTIONS)

        errors: list[Exception] = []
        stop = threading.Event()

        def on_error(exc):
            errors.append(exc)
            stop.set()  # 避免错误风暴，收到第一个错误即停

        thread = threading.Thread(
            target=worker.run_loop,
            kwargs={"stop_event": stop, "on_error": on_error, "interval_s": 0.005},
        )
        thread.start()
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        assert len(errors) == 1
        assert "boom" in str(errors[0])
