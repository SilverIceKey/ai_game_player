"""ExplorationSkill：包装 CoverageExplorer（M3 计划 2.2——是包装不是重写）。

覆盖式探索内部逻辑（迟滞/航向承诺/卡住脱困/A* 归位）全部保留在
core/decision/navigation.py；本技能只负责生命周期与上下文接线：
- start 时从共享黑板取探索断点（脱战后归位继续漫游）
- 接敌由调度器抢占（interrupt → CombatSkill）
- scene=dead → NEEDS_HUMAN
"""
from __future__ import annotations

import math

from core.contracts import Action
from core.decision.navigation import CoverageExplorer
from core.perception.walkable import WalkableResult
from core.skills.base import SkillContext, SkillStatus, SkillTickResult


class ExplorationSkill:
    name = "exploration"
    priority = 10
    interruptible = True

    def __init__(self, explorer: CoverageExplorer):
        self.explorer = explorer
        self._return_target: tuple[float, float] | None = None

    def can_start(self, context: SkillContext) -> bool:
        return context.state.scene != "dead"

    def start(self, context: SkillContext) -> None:
        # 脱战断点恢复：CombatSkill 结束时放入共享黑板
        self._return_target = context.shared.get("explore_breakpoint")

    def tick(self, context: SkillContext) -> SkillTickResult:
        state = context.state
        if state.scene == "dead":
            return SkillTickResult(
                SkillStatus.NEEDS_HUMAN, Action("idle"), "死亡，停止输入等待人工"
            )
        raw = state.raw
        pose = tuple(raw.get("pose") or (0.0, 0.0, 0.0))

        target = self._return_target
        if target is not None and math.hypot(pose[0] - target[0], pose[1] - target[1]) <= (
            self.explorer.params.arrive_distance
        ):
            self._return_target = None
            context.shared.pop("explore_breakpoint", None)
            target = None

        walk = raw.get("walkable") or {}
        walkable = WalkableResult(
            float(walk.get("left", 0.0)),
            float(walk.get("center", 0.0)),
            float(walk.get("right", 0.0)),
            str(walk.get("suggestion", "straight")),
        )
        action = self.explorer.decide(pose, walkable, target)
        desc = "返回探索断点" if target is not None else "覆盖漫游"
        return SkillTickResult(SkillStatus.RUNNING, action, f"EXPLORE: {desc}")

    def interrupt(self, reason: str) -> None:
        pass  # 探索可随时被打断；CoverageExplorer 的位姿/栅格状态天然保留

    def dispose(self) -> None:
        pass
