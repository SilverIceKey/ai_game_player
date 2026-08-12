"""Episode 手动切分状态机（spec §21）。

MVP 不做自动 Episode 检测（spec §21：不要为了全自动提前增加不稳定感知模块），
由人工（safety.episode_key，默认 F9）触发 START/STOP。

边界行为（显式约定）：
- 未 start 就 stop：忽略，记 warning，返回 None。
- 进行中重复 start：忽略，记 warning，当前 episode 继续。
- stop 时间早于 start 时间：抛 ValueError（时间倒流说明上游时钟用错，不静默）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from capture.action import SOURCE_HUMAN

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EpisodeMeta:
    """一段已结束 episode 的元信息（供 EpisodeStoreWriter 落盘）。"""

    episode_id: int
    start_us: int
    end_us: int
    source: str = SOURCE_HUMAN


class EpisodeTracker:
    """手动 START/STOP 状态机。"""

    def __init__(self) -> None:
        self._next_id = 0
        self._active_start_us: int | None = None
        self._active_source: str = SOURCE_HUMAN

    @property
    def is_active(self) -> bool:
        return self._active_start_us is not None

    @property
    def current_episode_id(self) -> int | None:
        """进行中的 episode_id；未在进行时为 None。"""
        return self._next_id if self.is_active else None

    def start(self, timestamp_us: int, source: str = SOURCE_HUMAN) -> bool:
        """START EPISODE。已在进行中则忽略并返回 False。"""
        if self.is_active:
            logger.warning(
                "episode %d 进行中，忽略重复 start (timestamp_us=%d)",
                self._next_id,
                timestamp_us,
            )
            return False
        self._active_start_us = int(timestamp_us)
        self._active_source = source
        return True

    def stop(self, timestamp_us: int) -> EpisodeMeta | None:
        """STOP EPISODE，返回元信息；未在进行时忽略并返回 None。"""
        if not self.is_active:
            logger.warning("没有进行中的 episode，忽略 stop (timestamp_us=%d)", timestamp_us)
            return None
        assert self._active_start_us is not None
        if timestamp_us < self._active_start_us:
            raise ValueError(
                f"stop 时间 {timestamp_us} 早于 start 时间 {self._active_start_us}，时间倒流"
            )
        meta = EpisodeMeta(
            episode_id=self._next_id,
            start_us=self._active_start_us,
            end_us=int(timestamp_us),
            source=self._active_source,
        )
        self._active_start_us = None
        self._next_id += 1
        return meta
