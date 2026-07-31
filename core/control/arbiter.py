"""ActionArbiter：动作仲裁（M3 计划 2.3；规格书 §9）。

优先级：人工急停 > 失焦保护 > 反射闪避 > 技能动作（ActionSource 数值大者优先）。
动作带有效期（默认 500ms，control.action_ttl_ms 可配）：过期动作丢弃不执行。
急停/失焦状态下一切动作被拒绝。每次仲裁结果写日志（谁胜出 / 谁被丢弃及原因）。
"""
from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime

from core.contracts import Action


class ActionSource(enum.IntEnum):
    """动作来源优先级（数值大 = 优先）。"""

    SKILL = 10  # 技能动作（探索/战斗）
    REFLEX = 20  # 反射动作（如紧急闪避）
    FOCUS_GUARD = 30  # 失焦保护
    EMERGENCY = 40  # 人工急停


@dataclass(frozen=True)
class CandidateAction:
    action: Action
    source: ActionSource
    issued_at: float  # time.monotonic() 时钟
    ttl_ms: float | None = None  # None = 用仲裁器默认值（control.action_ttl_ms）

    def expired(self, now: float, default_ttl_ms: float) -> bool:
        ttl = self.ttl_ms if self.ttl_ms is not None else default_ttl_ms
        return (now - self.issued_at) * 1000.0 > ttl


@dataclass(frozen=True)
class ArbiterDecision:
    action: Action | None  # None = 本 tick 无动作可执行
    winner: ActionSource | None
    dropped: tuple[tuple[ActionSource, str], ...] = ()  # (来源, 丢弃原因)


class ActionArbiter:
    def __init__(
        self,
        default_ttl_ms: float = 500.0,
        logger: logging.Logger | None = None,
        clock=time.monotonic,
    ):
        self.default_ttl_ms = float(default_ttl_ms)
        self._log = logger or logging.getLogger("auto_player")
        self._clock = clock

    def decide(
        self,
        candidates: list[CandidateAction],
        blocked: bool = False,
        blocked_reason: str = "",
    ) -> ArbiterDecision:
        """仲裁一批候选动作。blocked=True（急停/失焦）时拒绝一切动作。"""
        now = self._clock()
        dropped: list[tuple[ActionSource, str]] = []

        if blocked:
            dropped.extend((c.source, f"blocked:{blocked_reason}") for c in candidates)
            decision = ArbiterDecision(None, None, tuple(dropped))
            self._log_decision(decision)
            return decision

        alive: list[CandidateAction] = []
        for c in candidates:
            if c.expired(now, self.default_ttl_ms):
                dropped.append((c.source, "expired"))
            else:
                alive.append(c)

        winner: CandidateAction | None = None
        for c in alive:
            # 优先级高者胜；同级取签发最新者
            if winner is None or (c.source, c.issued_at) > (winner.source, winner.issued_at):
                if winner is not None:
                    dropped.append((winner.source, "preempted"))
                winner = c
            else:
                dropped.append((c.source, "preempted"))

        decision = ArbiterDecision(
            winner.action if winner else None,
            winner.source if winner else None,
            tuple(dropped),
        )
        self._log_decision(decision)
        return decision

    def _log_decision(self, decision: ArbiterDecision) -> None:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        if decision.winner is None:
            self._log.info(
                "[%s] arbiter winner=none dropped=%s", ts, self._fmt_dropped(decision.dropped)
            )
        else:
            self._log.info(
                "[%s] arbiter winner=%s action=%s dropped=%s",
                ts, decision.winner.name.lower(), decision.action.name if decision.action else "?",
                self._fmt_dropped(decision.dropped),
            )

    @staticmethod
    def _fmt_dropped(dropped: tuple[tuple[ActionSource, str], ...]) -> str:
        if not dropped:
            return "-"
        return ",".join(f"{src.name.lower()}:{reason}" for src, reason in dropped)
