"""技能抽象（M3 计划 2.1；规格书 5.9 的 Python 化）。

技能在生命周期内连续控制（治"每 tick 重新决策导致动作僵硬"），
由 SkillScheduler 调度与抢占；动作经 ActionArbiter 仲裁后执行。
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from core.contracts import Action, GameState


class SkillStatus(enum.Enum):
    """技能 tick 五态。"""

    RUNNING = "RUNNING"  # 继续执行，给出候选动作
    SUCCEEDED = "SUCCEEDED"  # 目标达成，交还控制权（如脱战）
    FAILED = "FAILED"  # 失败终止
    NEEDS_REPLAN = "NEEDS_REPLAN"  # 需要重新规划（调度器换技能）
    NEEDS_HUMAN = "NEEDS_HUMAN"  # 需要人工介入（如死亡），停止输入


@dataclass(frozen=True)
class SkillTickResult:
    status: SkillStatus
    action: Action | None = None  # RUNNING 时的候选动作（交仲裁）
    reason: str = ""  # 意图/结束原因（日志用）


@dataclass
class SkillContext:
    """技能 tick 上下文：当前感知状态 + 跨技能共享黑板（探索断点等）。"""

    state: GameState
    tick: int
    shared: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class GameSkill(Protocol):
    """游戏技能协议：生命周期 start → tick* → (SUCCEEDED|FAILED|...) / interrupt → dispose。"""

    name: str
    priority: int  # 数值大 = 优先级高（战斗 > 探索）
    interruptible: bool

    def can_start(self, context: SkillContext) -> bool: ...

    def start(self, context: SkillContext) -> None: ...

    def tick(self, context: SkillContext) -> SkillTickResult: ...

    def interrupt(self, reason: str) -> None: ...

    def dispose(self) -> None: ...
