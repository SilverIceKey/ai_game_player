"""定长环形缓冲（spec §30：History Ring Buffer；§8.1 Video History / §8.2 Action History）。

AUTOPILOT 闭环中，Capture Thread 推帧、Inference Worker 取时间窗，
两者并发访问，因此两个缓冲都用锁保证线程安全。
窗口语义：window(n)/recent(n) 返回最近 n 个元素（按时间升序），
不足 n 个时返回已有的全部（由调用方决定是否满足推理窗口要求）。
"""
from __future__ import annotations

import threading
from collections import deque

import numpy as np

from capture.action import ActionRecord, NormalizedAction


class FrameRingBuffer:
    """历史帧环形缓冲：元素为 (frame, timestamp_us)，容量固定、满则丢最旧。"""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError(f"capacity 必须为正整数: {capacity!r}")
        self._capacity = capacity
        self._frames: deque[tuple[np.ndarray, int]] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)

    def push(self, frame: np.ndarray, timestamp_us: int) -> None:
        with self._lock:
            self._frames.append((frame, timestamp_us))

    def latest(self) -> tuple[np.ndarray, int] | None:
        """最新一帧 (frame, timestamp_us)；空缓冲返回 None。"""
        with self._lock:
            if not self._frames:
                return None
            return self._frames[-1]

    def window(self, n: int) -> list[tuple[np.ndarray, int]]:
        """最近 n 个 (frame, timestamp_us)，按时间升序；不足 n 个返回已有全部。"""
        if n <= 0:
            return []
        with self._lock:
            items = list(self._frames)
        return items[-n:]


class ActionHistoryBuffer:
    """动作历史环形缓冲（spec §8.2：模型输入的 Action[t-m:t-1]）。"""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError(f"capacity 必须为正整数: {capacity!r}")
        self._capacity = capacity
        self._records: deque[ActionRecord] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def push(self, record: ActionRecord) -> None:
        with self._lock:
            self._records.append(record)

    def recent(self, n: int) -> list[NormalizedAction]:
        """最近 n 个动作，按时间升序；不足 n 个返回已有全部。"""
        if n <= 0:
            return []
        with self._lock:
            records = list(self._records)
        return [r.action for r in records[-n:]]
