"""Inference Worker（spec §30 线程结构；§33 Runtime 日志字段；§31 延迟目标）。

从 FrameRingBuffer 取 Video History 窗口、从 ActionHistoryBuffer 取动作历史，
调用注入的 policy 做一次推理，产出 ActionChunk 与统计字段。

policy 为鸭子类型（禁止 import model/ 包，避免依赖方向倒置）：
    policy.predict(frames: list[np.ndarray], action_history: list[NormalizedAction]) -> ActionChunk
    （注入 audio_buffer 时追加第三个位置参数 audio_pcm，见下）

音频（spec §8.5）：注入 audio_buffer 时，infer_once 额外切出与 Video History
对齐的过去窗口 PCM 传给 policy（mel 特征提取在 policy 内部，与训练共用
model/audio_features.py）。

明确语义：窗口帧数不足 history_frames 时 infer_once 返回 None（不推理、
不产生任何输出），由调用方决定等待或跳过。
"""
from __future__ import annotations

import threading
from collections.abc import Callable

from capture.action import ActionChunk
from capture.clock import now_us
from config import PredictionConfig
from runtime.ring_buffer import ActionHistoryBuffer, AudioRingBuffer, FrameRingBuffer


class InferenceWorker:
    """单次推理 + 可选后台循环（供 app 层以独立线程运行）。"""

    def __init__(
        self,
        policy: object,
        frame_buffer: FrameRingBuffer,
        action_history: ActionHistoryBuffer,
        history_frames: int,
        history_actions: int,
        prediction: PredictionConfig | None = None,
        audio_buffer: AudioRingBuffer | None = None,
        audio_window_us: int = 0,
    ):
        if history_frames <= 0:
            raise ValueError(f"history_frames 必须为正整数: {history_frames!r}")
        if audio_buffer is not None and audio_window_us <= 0:
            raise ValueError(f"注入 audio_buffer 时 audio_window_us 必须为正: {audio_window_us!r}")
        self._policy = policy
        self._frame_buffer = frame_buffer
        self._action_history = action_history
        self._history_frames = history_frames
        self._history_actions = history_actions
        self._prediction = prediction or PredictionConfig()
        self._audio_buffer = audio_buffer
        self._audio_window_us = audio_window_us

    def infer_once(self, now_us_value: int) -> tuple[ActionChunk, dict] | None:
        """取窗口 → policy.predict → 记录耗时。窗口不足返回 None。

        stats 字段（spec §33）：timestamp_us / model_version / frame_age_ms /
        queue_delay_ms / inference_ms。耗时用真实时钟测量（时间戳入参仅用于
        标记本次推理与计算帧龄，保证可注入可测）。
        """
        window = self._frame_buffer.window(self._history_frames)
        if len(window) < self._history_frames:
            return None

        t0 = now_us()
        frames = [frame for frame, _ in window]
        newest_ts = window[-1][1]
        history = self._action_history.recent(self._history_actions)
        t1 = now_us()
        if self._audio_buffer is not None:
            audio_pcm = self._audio_buffer.window(newest_ts - self._audio_window_us, self._audio_window_us)
            chunk = self._policy.predict(frames, history, audio_pcm)
        else:
            chunk = self._policy.predict(frames, history)
        t2 = now_us()

        stats = {
            "timestamp_us": now_us_value,
            "model_version": chunk.model_version,
            "frame_age_ms": (now_us_value - newest_ts) / 1000.0,
            "queue_delay_ms": (t1 - t0) / 1000.0,
            "inference_ms": (t2 - t1) / 1000.0,
        }
        return chunk, stats

    def run_loop(
        self,
        stop_event: threading.Event,
        on_result: Callable[[ActionChunk, dict], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
        interval_s: float | None = None,
    ) -> None:
        """推理后台循环（spec §30：inference 独立线程，禁止与 capture/input 互相阻塞）。

        默认节奏 = chunk 时长（action_step_ms × future_action_steps，spec §15），
        与 ActionScheduler 派发完一个 chunk 的时间对齐。用 stop_event.wait 睡眠，
        置位后立即退出。窗口不足时本轮跳过，不视为错误。
        """
        if interval_s is None:
            interval_s = (
                self._prediction.action_step_ms * self._prediction.future_action_steps / 1000.0
            )
        while not stop_event.is_set():
            loop_start = now_us()
            try:
                result = self.infer_once(loop_start)
            except Exception as exc:
                if on_error is not None:
                    on_error(exc)
                else:
                    raise
            else:
                if result is not None and on_result is not None:
                    on_result(result[0], result[1])
            elapsed_s = (now_us() - loop_start) / 1_000_000.0
            stop_event.wait(max(0.0, interval_s - elapsed_s))
