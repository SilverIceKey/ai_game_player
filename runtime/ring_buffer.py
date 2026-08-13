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

    def recent_records(self, n: int) -> list[ActionRecord]:
        """最近 n 条 ActionRecord（带时间戳，spec §16 token 年龄用），按时间升序。"""
        if n <= 0:
            return []
        with self._lock:
            records = list(self._records)
        return records[-n:]


class AudioRingBuffer:
    """音频环形缓冲（spec §8.5 Audio History）：元素为 (chunk_start_us, pcm_f32 mono)。

    容量按采样点数控制，满则丢最旧的整块；window() 按时间窗切片，
    缓冲覆盖不到的部分零填充（与训练侧 load_audio_window 的语义一致）。
    """

    def __init__(self, capacity_seconds: float, sample_rate: int):
        if capacity_seconds <= 0:
            raise ValueError(f"capacity_seconds 必须为正数: {capacity_seconds!r}")
        self._sr = int(sample_rate)
        self._max_samples = int(capacity_seconds * self._sr)
        self._chunks: deque[tuple[int, np.ndarray]] = deque()
        self._total = 0
        self._lock = threading.Lock()

    def push(self, chunk_start_us: int, pcm: np.ndarray) -> None:
        pcm = np.asarray(pcm, dtype=np.float32)
        with self._lock:
            self._chunks.append((int(chunk_start_us), pcm))
            self._total += len(pcm)
            while self._total > self._max_samples and len(self._chunks) > 1:
                _, old = self._chunks.popleft()
                self._total -= len(old)

    def window(self, start_us: int, duration_us: int) -> np.ndarray:
        """切出 [start_us, start_us+duration_us) 的 float32 mono，未覆盖部分零填充。"""
        n = max(1, int(round(duration_us * self._sr / 1e6)))
        with self._lock:
            chunks = list(self._chunks)
        out = np.zeros(n, dtype=np.float32)
        end_us = start_us + duration_us
        for chunk_start, pcm in chunks:
            chunk_end = chunk_start + int(len(pcm) * 1e6 / self._sr)
            lo, hi = max(start_us, chunk_start), min(end_us, chunk_end)
            if hi <= lo:
                continue
            src_lo = int((lo - chunk_start) * self._sr / 1e6)
            src_hi = int((hi - chunk_start) * self._sr / 1e6)
            dst_lo = int((lo - start_us) * self._sr / 1e6)
            seg = pcm[src_lo:src_hi]
            out[dst_lo : dst_lo + len(seg)] = seg
        return out
