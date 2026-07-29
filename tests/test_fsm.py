"""战斗 FSM 状态转移测试（含 dead 分支）+ 通用状态机引擎测试。"""
from core.contracts import GameState
from core.decision.fsm import StateMachine
from core.decision.navigation import CoverageExplorer
from core.navigation.grid_map import OccupancyGrid
from games.wukong.adapter import WukongConfig
from games.wukong.combat import CombatDecision

WALKABLE = {"left": 0.5, "center": 0.8, "right": 0.5, "suggestion": "straight"}


def _make_decision() -> tuple[CombatDecision, WukongConfig]:
    cfg = WukongConfig.load("configs/wukong.yaml")
    grid = OccupancyGrid(cfg.exploration.grid_size_m, cfg.exploration.grid_resolution)
    return CombatDecision(cfg, CoverageExplorer(grid, cfg.exploration), grid), cfg


def _state(scene="explore", hp=1.0, stamina=1.0, gourd=True, enemy=None, pose=(0.0, 0.0, 0.0)):
    present = enemy is not None
    return GameState(
        timestamp=0.0,
        scene=scene,
        raw={
            "hp_ratio": hp,
            "stamina_ratio": stamina,
            "gourd_available": gourd,
            "enemy_hp_ratio": enemy,
            "enemy_present": present,
            "in_combat": present,
            "pose": pose,
            "walkable": dict(WALKABLE),
        },
    )


def _combat(enemy=0.9, **kwargs) -> GameState:
    return _state(scene="combat", enemy=enemy, **kwargs)


def _drive_to_combat(d: CombatDecision, cfg: WukongConfig) -> None:
    d.decide(_combat())  # → ENGAGE
    for _ in range(cfg.combat.engage_approach_ticks + 2):
        if d.state_name == "COMBAT":
            break
        d.decide(_combat())
    assert d.state_name == "COMBAT"


def test_explore_roaming():
    d, _ = _make_decision()
    action = d.decide(_state())
    assert d.state_name == "EXPLORE"
    assert action.name in ("move", "turn", "lock_on")
    assert d.intent.startswith("EXPLORE:")


def test_engage_then_combat():
    d, cfg = _make_decision()
    d.decide(_state())
    action = d.decide(_combat())
    assert d.state_name == "ENGAGE"
    assert action.name == "lock_on"  # 接敌首 tick 锁定
    action = d.decide(_combat())
    assert d.state_name == "ENGAGE" and action.name == "move"  # 接近中
    for _ in range(cfg.combat.engage_approach_ticks + 2):
        if d.state_name == "COMBAT":
            break
        action = d.decide(_combat())
    assert d.state_name == "COMBAT"
    assert action.name == "light_attack"


def test_heal_branch():
    d, cfg = _make_decision()
    _drive_to_combat(d, cfg)
    action = d.decide(_combat(hp=cfg.combat.heal_hp_threshold - 0.05, gourd=True))
    assert d.state_name == "HEAL"
    assert action.name == "heal"
    # 喝药单步后回 COMBAT（葫芦耗尽不再触发 HEAL）
    action = d.decide(_combat(hp=cfg.combat.heal_hp_threshold - 0.05, gourd=False))
    assert d.state_name == "COMBAT"
    assert action.name == "light_attack"


def test_dodge_on_hit():
    d, cfg = _make_decision()
    _drive_to_combat(d, cfg)
    d.decide(_combat(hp=1.0))
    action = d.decide(_combat(hp=1.0 - cfg.combat.dodge_on_hit_drop - 0.01))
    assert d.state_name == "DODGE"
    assert action.name == "dodge"
    action = d.decide(_combat(hp=0.8))
    assert d.state_name == "COMBAT"
    assert action.name == "light_attack"


def test_periodic_dodge():
    d, cfg = _make_decision()
    _drive_to_combat(d, cfg)
    seen_dodge = False
    for _ in range(cfg.combat.dodge_interval_ticks + 3):
        d.decide(_combat(hp=1.0))
        if d.state_name == "DODGE":
            seen_dodge = True
            break
    assert seen_dodge, "固定节奏闪避未触发"


def test_enemy_lost_loot_wait_then_explore():
    d, cfg = _make_decision()
    _drive_to_combat(d, cfg)
    action = None
    for _ in range(cfg.combat.enemy_lost_ticks + cfg.combat.dodge_interval_ticks):
        action = d.decide(_state())  # 敌方血条消失
        if d.state_name == "LOOT_WAIT":
            break
    assert d.state_name == "LOOT_WAIT"
    assert action.name == "idle"
    for _ in range(cfg.combat.loot_wait_ticks + 1):
        action = d.decide(_state())
        if d.state_name == "EXPLORE":
            break
    assert d.state_name == "EXPLORE"
    assert action.name in ("move", "turn", "lock_on")


def test_dead_from_combat_stops_input():
    d, cfg = _make_decision()
    _drive_to_combat(d, cfg)
    action = d.decide(_state(scene="dead", hp=0.0))
    assert d.state_name == "DEAD"
    assert action.name == "idle"
    # DEAD 为终态：持续停止输入
    action = d.decide(_state(scene="dead", hp=0.0))
    assert d.state_name == "DEAD"
    assert action.name == "idle"


def test_dead_from_explore():
    d, _ = _make_decision()
    action = d.decide(_state(scene="dead", hp=0.0))
    assert d.state_name == "DEAD"
    assert action.name == "idle"
    assert d.intent.startswith("DEAD:")


def test_generic_state_machine():
    m = StateMachine("A")
    m.add_transition("A", "B", lambda s: bool(s.raw.get("go")))
    m.add_global_transition("Z", lambda s: s.scene == "dead")

    assert not m.update(GameState(timestamp=0.0))
    assert m.current == "A"
    assert m.update(GameState(timestamp=0.0, raw={"go": True}))
    assert m.current == "B"
    # 全局转移优先且跨状态生效
    assert m.update(GameState(timestamp=0.0, scene="dead", raw={"go": True}))
    assert m.current == "Z"
    # 终态：无转出定义则保持
    assert not m.update(GameState(timestamp=0.0, scene="dead"))
    assert m.current == "Z"
