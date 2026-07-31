"""CombatSkill：包装 CombatDecision 战斗 FSM（M3 计划 2.2——是包装不是重写）。

ENGAGE/COMBAT/HEAL/DODGE/LOOT_WAIT 转移与动作映射全部保留在
games/wukong/combat.py；本技能只负责生命周期与控制权交接：
- FSM 回到 EXPLORE（脱战）→ SUCCEEDED，探索断点移交共享黑板
- FSM 进入 DEAD → NEEDS_HUMAN（停止输入，等待人工）
- 战斗进行中不可被普通抢占（急停/失焦由安全层在仲裁侧阻断，不走 interrupt）
"""
from __future__ import annotations

from core.contracts import Action
from core.skills.base import SkillContext, SkillStatus, SkillTickResult
from games.wukong.combat import CombatDecision


class CombatSkill:
    name = "combat"
    priority = 20
    interruptible = False

    def __init__(self, decision: CombatDecision):
        self.decision = decision

    def can_start(self, context: SkillContext) -> bool:
        return bool(context.state.raw.get("enemy_present"))

    def start(self, context: SkillContext) -> None:
        pass  # FSM 处于 EXPLORE；首个 tick 的 decide 会完成 EXPLORE→ENGAGE（记录断点）

    def tick(self, context: SkillContext) -> SkillTickResult:
        action = self.decision.decide(context.state)
        fsm = self.decision.state_name
        if fsm == "DEAD":
            return SkillTickResult(
                SkillStatus.NEEDS_HUMAN, Action("idle"), "死亡，停止输入等待人工"
            )
        if fsm == "EXPLORE":
            # 脱战：探索断点移交调度器共享黑板，交还控制权
            context.shared["explore_breakpoint"] = self.decision.return_target
            self.decision.clear_return_target()
            return SkillTickResult(SkillStatus.SUCCEEDED, None, "脱战，交还探索")
        return SkillTickResult(SkillStatus.RUNNING, action, self.decision.intent)

    def interrupt(self, reason: str) -> None:
        pass  # 战斗不被普通抢占；急停/失焦由安全层处理

    def dispose(self) -> None:
        pass
