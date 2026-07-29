"""黑神话悟空战斗决策：战斗 FSM（计划文档 3.3 节）。

状态机：EXPLORE → ENGAGE → COMBAT ⇄ HEAL/DODGE → LOOT_WAIT → EXPLORE，
任意状态 scene=dead → DEAD（停止输入，等待人工）。
脱战后回到战斗开始时的探索断点继续覆盖漫游。
"""
from __future__ import annotations

import math

from core.contracts import Action, GameState
from core.decision.fsm import StateMachine
from core.decision.navigation import CoverageExplorer
from core.navigation.grid_map import OccupancyGrid
from core.perception.walkable import WalkableResult
from games.wukong.adapter import WukongConfig


class CombatDecision:
    """实现 core.decision.base.DecisionEngine 契约（全自动模式返回 Action）。"""

    def __init__(self, config: WukongConfig, explorer: CoverageExplorer, grid: OccupancyGrid):
        self.config = config
        self.explorer = explorer
        self.grid = grid
        self.intent: str = ""  # 最近一次决策意图（日志用）

        self._machine = StateMachine("EXPLORE")
        self._state_ticks = 0  # 当前状态持续的 tick 数
        self._enemy_lost = 0  # 敌方血条连续消失 tick 数
        self._prev_hp: float | None = None
        self._hit_taken = False
        self._return_target: tuple[float, float] | None = None  # 探索断点

        c = config.combat
        m = self._machine
        m.add_global_transition("DEAD", lambda s: s.scene == "dead")
        m.add_transition("EXPLORE", "ENGAGE", lambda s: bool(s.raw.get("enemy_present")))
        m.add_transition(
            "ENGAGE", "COMBAT",
            lambda s: bool(s.raw.get("enemy_present")) and self._state_ticks >= c.engage_approach_ticks,
        )
        m.add_transition(
            "ENGAGE", "EXPLORE",
            lambda s: not s.raw.get("enemy_present") and self._state_ticks >= c.enemy_lost_ticks,
        )
        m.add_transition(
            "COMBAT", "HEAL",
            lambda s: float(s.raw.get("hp_ratio") or 0.0) < c.heal_hp_threshold
            and bool(s.raw.get("gourd_available")),
        )
        m.add_transition(
            "COMBAT", "DODGE",
            lambda s: self._hit_taken
            or (self._state_ticks > 0 and self._state_ticks % c.dodge_interval_ticks == 0),
        )
        m.add_transition("COMBAT", "LOOT_WAIT", lambda s: self._enemy_lost >= c.enemy_lost_ticks)
        m.add_transition("HEAL", "COMBAT", lambda s: self._state_ticks >= 1)
        m.add_transition("DODGE", "COMBAT", lambda s: self._state_ticks >= 1)
        m.add_transition("LOOT_WAIT", "EXPLORE", lambda s: self._state_ticks >= c.loot_wait_ticks)

    @property
    def state_name(self) -> str:
        return self._machine.current

    def decide(self, state: GameState) -> Action:
        raw = state.raw
        pose = tuple(raw.get("pose") or (0.0, 0.0, 0.0))
        c = self.config.combat

        # 计数器与受击检测（转移求值前更新）
        self._enemy_lost = 0 if raw.get("enemy_present") else self._enemy_lost + 1
        hp = raw.get("hp_ratio")
        if (
            self._machine.current == "COMBAT"
            and self._prev_hp is not None
            and hp is not None
            and self._prev_hp - float(hp) >= c.dodge_on_hit_drop
        ):
            self._hit_taken = True

        prev_state = self._machine.current
        self._state_ticks += 1
        self._machine.update(state)
        if self._machine.current != prev_state:
            self._state_ticks = 0
            self._on_enter(self._machine.current, pose)
        self._prev_hp = float(hp) if hp is not None else None
        self._hit_taken = False

        action, desc = self._act(pose, raw)
        self.intent = f"{self._machine.current}: {desc}"
        return action

    # ---------- 内部 ----------

    def _on_enter(self, state_name: str, pose: tuple[float, float, float]) -> None:
        if state_name == "ENGAGE":
            # 记录探索断点：脱战后回到这里继续漫游
            self._return_target = (pose[0], pose[1])

    def _act(self, pose: tuple[float, float, float], raw: dict) -> tuple[Action, str]:
        name = self._machine.current
        if name == "DEAD":
            return Action("idle"), "死亡，停止输入等待人工"
        if name == "ENGAGE":
            if self._state_ticks == 0:
                return Action("lock_on"), "锁定目标并接近"
            return Action("move", {"direction": "forward"}), "锁定目标并接近"
        if name == "COMBAT":
            return Action("light_attack"), "持续输出"
        if name == "HEAL":
            return Action("heal"), "低血喝药"
        if name == "DODGE":
            return Action("dodge"), "闪避"
        if name == "LOOT_WAIT":
            return Action("idle"), "等待掉落/脱战"

        # EXPLORE：覆盖漫游；有探索断点先归位
        target = self._return_target
        if target is not None and math.hypot(pose[0] - target[0], pose[1] - target[1]) <= (
            self.explorer.params.arrive_distance
        ):
            self._return_target = None
            target = None
        walk = raw.get("walkable") or {}
        walkable = WalkableResult(
            float(walk.get("left", 0.0)),
            float(walk.get("center", 0.0)),
            float(walk.get("right", 0.0)),
            str(walk.get("suggestion", "straight")),
        )
        action = self.explorer.decide(pose, walkable, target)
        return action, ("返回探索断点" if target is not None else "覆盖漫游")
