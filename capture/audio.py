"""音频采集（spec §8.5）：系统音频输出回录（WASAPI loopback）。

- 生产实现 `SoundcardLoopbackCapture` 依赖 soundcard 库（仅 Windows），延迟导入；
  测试通过注入假后端覆盖同一路径。
- 输出统一为 mono float32 [-1, 1]，采样率来自 config.AudioConfig。
- 时间戳（spec §11）：块到达时刻用 capture.clock.now_us() 打点，
  块起始时间 = 到达时刻 − 块时长。音频是秒级线索，此近似足够（spec §8.5）。
"""
from __future__ import annotations

import threading
from typing import Callable, Protocol

import numpy as np

from capture.clock import now_us


class AudioCapture(Protocol):
    """阻塞式音频块读取接口（生产: WASAPI loopback；测试: 注入假后端）。"""

    sample_rate: int

    def open(self) -> None: ...

    def read(self, num_frames: int) -> np.ndarray:
        """阻塞读取 num_frames 个采样点，返回 float32 mono (num_frames,)。"""
        ...

    def close(self) -> None: ...


class SoundcardLoopbackCapture:
    """默认扬声器对应的 loopback 设备回录（不需要麦克风权限）。"""

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = int(sample_rate)
        self._recorder: object | None = None

    def open(self) -> None:
        import soundcard as sc  # 延迟导入：仅 Windows 实机需要

        speaker = sc.default_speaker()
        loopback = sc.get_microphone(speaker.name, include_loopback=True)
        self._recorder = loopback.recorder(samplerate=self.sample_rate, channels=1)
        self._recorder.__enter__()  # type: ignore[union-attr]

    def read(self, num_frames: int) -> np.ndarray:
        if self._recorder is None:
            raise RuntimeError("SoundcardLoopbackCapture 未 open")
        data = self._recorder.record(numframes=num_frames)  # type: ignore[union-attr]
        mono = data[:, 0] if data.ndim == 2 else data
        return np.asarray(mono, dtype=np.float32)

    def close(self) -> None:
        if self._recorder is not None:
            self._recorder.__exit__(None, None, None)  # type: ignore[union-attr]
            self._recorder = None


# 回调签名：(块起始时间戳 us, float32 mono PCM)
AudioChunkHandler = Callable[[int, np.ndarray], None]


def run_capture_loop(
    capture: AudioCapture,
    on_chunk: AudioChunkHandler,
    stop_event: threading.Event,
    chunk_ms: int = 100,
) -> None:
    """采集线程主循环：阻塞读块 → 打点 → 回调（OBSERVE_TRAIN/AUTOPILOT 共用）。

    音频块到达时刻打点，块起始时间 = 到达时刻 − 块时长（spec §8.5/§11）。
    """
    chunk_frames = max(1, int(capture.sample_rate * chunk_ms / 1000))
    capture.open()
    try:
        while not stop_event.is_set():
            pcm = capture.read(chunk_frames)
            end_us = now_us()
            start_us = end_us - int(len(pcm) * 1e6 / capture.sample_rate)
            on_chunk(start_us, pcm)
    finally:
        capture.close()
