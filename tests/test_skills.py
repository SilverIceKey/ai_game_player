"""M3 技能层测试：生命周期五态、接敌抢占、脱战断点恢复、死亡 NEEDS_HUMAN。"""
import pytest

from core.contracts import GameState
from core.decision.navigation import CoverageExplorer
from core.navigation.grid_map import OccupancyGrid
from core.skills.base import GameSkill, SkillContext, SkillStatus
from core.skills.combat import CombatSkill
from core.skills.exploration import ExplorationSkill
from core.skills.scheduler import SkillScheduler
from games.wukong.adapter import WukongConfig
from games.wukong.combat import CombatDecision

WALKABLE = {"left": 0.5, "center": 0.8, "right": 0.5, "suggestion": "straight"}


def _state(scene="explore", hp=1.0, gourd=True, enemy=None, pose=(0.0, 0.0, 0.0)):
    present = enemy is not None
    return GameState(
        timestamp=0.0,
        scene=scene,
        raw={
            "hp_ratio": hp, "stamina_ratio": 1.0, "gourd_available": gourd,
            "enemy_hp_ratio": enemy, "enemy_present": present, "in_combat": present,
            "pose": pose, "walkable": dict(WALKABLE),
        },
    )


def _make():
    cfg = WukongConfig.load("configs/wukong.yaml")
    grid = OccupancyGrid(cfg.exploration.grid_size_m, cfg.exploration.grid_resolution)
    explorer = CoverageExplorer(grid, cfg.exploration)
    decision = CombatDecision(cfg, explorer, grid)
    exploration = ExplorationSkill(explorer)
    combat = CombatSkill(decision)
    scheduler = SkillScheduler(exploration, combat)
    return scheduler, exploration, combat, decision, cfg


# ---------- 协议与生命周期 ----------

def test_skills_satisfy_protocol():
    _, exploration, combat, _, _ = _make()
    assert isinstance(exploration, GameSkill)
    assert isinstance(combat, GameSkill)
    assert combat.priority > exploration.priority
    assert exploration.interruptible and not combat.interruptible


def test_exploration_tick_running():
    _, exploration, _, _, _ = _make()
    ctx = SkillContext(state=_state(), tick=0, shared={})
    assert exploration.can_start(ctx)
    exploration.start(ctx)
    result = exploration.tick(ctx)
    assert result.status == SkillStatus.RUNNING
    assert result.action.name in ("move", "turn", "lock_on")
    assert result.reason.startswith("EXPLORE:")


def test_exploration_dead_needs_human():
    _, exploration, _, _, _ = _make()
    ctx = SkillContext(state=_state(scene="dead", hp=0.0), tick=0, shared={})
    assert not exploration.can_start(ctx)
    exploration.start(ctx)
    result = exploration.tick(ctx)
    assert result.status == SkillStatus.NEEDS_HUMAN
    assert result.action.name == "idle"


def test_combat_skill_succeeded_on_disengage():
    _, _, combat, decision, cfg = _make()
    ctx = SkillContext(state=_state(scene="combat", enemy=0.9, pose=(5.0, 5.0, 0.0)),
                       tick=0, shared={})
    assert combat.can_start(ctx)
    combat.start(ctx)
    result = combat.tick(ctx)
    assert result.status == SkillStatus.RUNNING
    assert result.action.name == "lock_on"  # ENGAGE 首 tick 锁定
    # 敌人消失（角色已移动离开断点）：ENGAGE 连续丢敌 → FSM 回 EXPLORE → SUCCEEDED
    for i in range(cfg.combat.enemy_lost_ticks + 2):
        ctx = SkillContext(state=_state(pose=(8.0, 8.0, 0.0)), tick=i + 1, shared=ctx.shared)
        result = combat.tick(ctx)
        if result.status == SkillStatus.SUCCEEDED:
            break
    assert result.status == SkillStatus.SUCCEEDED
    assert ctx.shared["explore_breakpoint"] == (5.0, 5.0)  # 断点移交黑板
    assert decision.return_target is None  # 已从 FSM 移交
    assert decision.state_name == "EXPLORE"


def test_combat_skill_dead_needs_human():
    _, _, combat, _, _ = _make()
    shared: dict = {}
    ctx = SkillContext(state=_state(scene="combat", enemy=0.9), tick=0, shared=shared)
    combat.start(ctx)
    combat.tick(ctx)
    result = combat.tick(SkillContext(state=_state(scene="dead", hp=0.0), tick=1, shared=shared))
    assert result.status == SkillStatus.NEEDS_HUMAN
    assert result.action.name == "idle"


# ---------- 调度器：抢占与恢复 ----------

def test_scheduler_preempt_on_engage():
    scheduler, _, _, _, _ = _make()
    action, intent, skill = scheduler.step(_state(), 0)
    assert skill == "exploration"
    assert action.name in ("move", "turn", "lock_on")
    # 接敌 → 抢占
    action, intent, skill = scheduler.step(_state(scene="combat", enemy=0.9), 1)
    assert skill == "combat"
    assert action.name == "lock_on"
    assert intent.startswith("ENGAGE:")


def test_scheduler_disengage_returns_to_exploration_with_breakpoint():
    scheduler, _, _, _, cfg = _make()
    scheduler.step(_state(), 0)
    combat_pose = (5.0, 5.0, 0.0)
    scheduler.step(_state(scene="combat", enemy=0.9, pose=combat_pose), 1)
    # 脱战：敌人消失（角色已离开断点）直至 SUCCEEDED → 回到探索
    skill = "combat"
    tick = 2
    while skill == "combat" and tick < cfg.combat.enemy_lost_ticks + 10:
        _, _, skill = scheduler.step(_state(pose=(8.0, 8.0, 0.0)), tick)
        tick += 1
    assert skill == "exploration"
    # 远离断点继续探索：应提示返回探索断点
    action, intent, skill = scheduler.step(_state(pose=(0.0, 0.0, 0.0)), tick)
    assert skill == "exploration"
    assert "返回探索断点" in intent
    # 回到断点位置 → 断点清除，恢复覆盖漫游
    _, intent, _ = scheduler.step(_state(pose=(5.0, 5.0, 0.0)), tick + 1)
    assert "覆盖漫游" in intent


def test_scheduler_needs_human_latches():
    scheduler, _, _, _, _ = _make()
    scheduler.step(_state(), 0)
    action, intent, skill = scheduler.step(_state(scene="dead", hp=0.0), 1)
    assert action.name == "idle"
    assert "NEEDS_HUMAN" in intent
    # 闩锁：后续 tick 即使画面恢复也保持停止输入，等待人工
    action, intent, skill = scheduler.step(_state(), 2)
    assert action.name == "idle"
    assert "NEEDS_HUMAN" in intent
    assert skill == "none"
