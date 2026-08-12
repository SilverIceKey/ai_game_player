"""Action Scheduler（spec §15 Action Chunking；§26/§40：接管/急停时清空）。

把一次推理输出的 ActionChunk 按 step_ms 节奏逐步吐出：
chunk 提交后立即吐出第 0 步，此后每过 step_ms 吐出下一步；
时间跨过多个 step（推理/调度抖动）时跳过中间步、直接取当前应执行的步，
保证动作与真实时间对齐而不是慢放。

submit_chunk 替换当前 chunk（旧 chunk 未执行的步直接丢弃，spec §15 一次推理
覆盖未来一段，新推理到达即作废旧计划）。clear() 用于 Human Override（§26）
与 Dead Man Switch（§40）：清空队列、不再吐出任何动作。

纯逻辑模块：时间由调用方注入（now_us），不做任何 I/O，可单测。
"""
from __future__ import annotations

from capture.action import ActionChunk, NormalizedAction


class ActionScheduler:
    def __init__(self, step_ms: float):
        if step_ms <= 0:
            raise ValueError(f"step_ms 必须为正数: {step_ms!r}")
        # 派发节奏以构造参数为准；调用方应保证与 chunk.step_ms 一致
        # （两者同源：prediction.action_step_ms）
        self._step_us = int(step_ms * 1000)
        self._chunk: ActionChunk | None = None
        self._start_us: int | None = None  # 当前 chunk 第 0 步的派发时刻
        self._index = 0  # 下一个待派发步下标

    @property
    def has_chunk(self) -> bool:
        return self._chunk is not None

    def submit_chunk(self, chunk: ActionChunk) -> None:
        """替换当前 chunk；首个动作在下一次 due_action 调用时立即派发。"""
        self._chunk = chunk
        self._start_us = None
        self._index = 0

    def clear(self) -> None:
        """清空当前 chunk（§26 接管 / §40 急停必须调用）。"""
        self._chunk = None
        self._start_us = None
        self._index = 0

    def due_action(self, now_us: int) -> NormalizedAction | None:
        """返回当前时刻应执行的动作；未到下一个 step 或无 chunk 返回 None。"""
        chunk = self._chunk
        if chunk is None:
            return None
        if self._start_us is None:
            self._start_us = now_us
            self._index = 1
            return chunk.actions[0]

        due_index = (now_us - self._start_us) // self._step_us
        if due_index < self._index:
            return None  # 下一个 step 还未到
        if due_index >= len(chunk.actions):
            self.clear()  # chunk 已派发完，等待下一次推理提交
            return None
        self._index = due_index + 1
        return chunk.actions[due_index]
