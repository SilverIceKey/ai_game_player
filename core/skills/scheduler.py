"""SkillScheduler：技能调度与抢占（M3 计划第 3 节主循环第 4 步）。

规则：
- 默认运行 ExplorationSkill
- 探索中 perceive 出 in_combat → interrupt 探索、start CombatSkill（抢占）
- CombatSkill SUCCEEDED（脱战）→ dispose，回到 ExplorationSkill（断点经共享黑板恢复）
- 任意技能 NEEDS_HUMAN（死亡等）→ 闩锁：停止输入，每 tick idle，等待人工
- NEEDS_REPLAN / FAILED → 简单重规划：按 can_start 重新选技能
"""
from __future__ import annotations

import logging

from core.contracts import Action, GameState
from core.skills.base import GameSkill, SkillContext, SkillStatus


class SkillScheduler:
    def __init__(self, exploration: GameSkill, combat: GameSkill, logger: logging.Logger | None = None):
        self.exploration = exploration
        self.combat = combat
        self.current: GameSkill = exploration
        self.needs_human = False
        self.shared: dict = {}
        self._log = logger or logging.getLogger("auto_player")
        self._started = False

    def step(self, state: GameState, tick: int) -> tuple[Action, str, str]:
        """推进一个 tick，返回 (候选动作, 意图, 当前技能名)。"""
        ctx = SkillContext(state=state, tick=tick, shared=self.shared)

        if self.needs_human:
            return Action("idle"), "NEEDS_HUMAN: 等待人工介入", "none"

        if not self._started:
            self.current.start(ctx)
            self._started = True
            self._log_transition("start", self.current.name)

        # 抢占：探索中接敌 → 战斗技能
        if (
            self.current is self.exploration
            and state.raw.get("in_combat")
            and self.combat.can_start(ctx)
        ):
            self.current.interrupt("接敌抢占")
            self._log_transition("preempt exploration→combat", "接敌")
            self.current = self.combat
            self.current.start(ctx)

        result = self.current.tick(ctx)

        if result.status == SkillStatus.RUNNING:
            return result.action or Action("idle"), result.reason, self.current.name

        if result.status == SkillStatus.NEEDS_HUMAN:
            self.needs_human = True
            self._log_transition("needs_human", result.reason)
            return Action("idle"), f"NEEDS_HUMAN: {result.reason}", self.current.name

        if result.status == SkillStatus.SUCCEEDED:
            self._log_transition(f"{self.current.name} succeeded", result.reason)
            self.current.dispose()
            self.current = self.exploration
            self.current.start(ctx)
            follow = self.current.tick(ctx)
            if follow.status == SkillStatus.RUNNING:
                return follow.action or Action("idle"), follow.reason, self.current.name
            return Action("idle"), follow.reason or result.reason, self.current.name

        # NEEDS_REPLAN / FAILED：按 can_start 重选技能
        self._log_transition(f"{self.current.name} {result.status.value}", result.reason)
        self.current.dispose()
        self.current = self.combat if self.combat.can_start(ctx) else self.exploration
        self.current.start(ctx)
        self._log_transition("replan", self.current.name)
        follow = self.current.tick(ctx)
        if follow.status == SkillStatus.RUNNING:
            return follow.action or Action("idle"), follow.reason, self.current.name
        return Action("idle"), follow.reason or result.reason, self.current.name

    def _log_transition(self, event: str, detail: str) -> None:
        from datetime import datetime

        self._log.info(
            "[%s] skill %s %s", datetime.now().strftime("%H:%M:%S.%f")[:-3], event, detail
        )
