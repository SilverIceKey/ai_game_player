"""逐 tick 日志格式测试（计划文档 3.4 节）。"""
import re

from apps.auto_player.main import format_tick
from core.contracts import Action, GameState


def _state() -> GameState:
    return GameState(
        timestamp=0.0,
        scene="combat",
        raw={
            "hp_ratio": 0.82,
            "stamina_ratio": 0.61,
            "mp_ratio": 0.66,
            "mp_visible": True,
            "enemy_hp_ratio": 0.45,
            "gourd_available": True,
            "pose": (12.3, 4.1, 1.518),  # ≈ 87°
        },
    )


def test_format_tick_three_lines():
    out = format_tick(_state(), "COMBAT: 持续输出", Action("light_attack"))
    lines = out.splitlines()
    assert len(lines) == 3
    assert re.match(
        r"^\[\d{2}:\d{2}:\d{2}\.\d{3}\] state scene=combat hp=0\.82 stamina=0\.61 "
        r"mp=0\.66 enemy_hp=0\.45 gourd=1 pos=\(12\.3,4\.1,87°\)$",
        lines[0],
    )
    assert lines[1].strip() == "intent COMBAT: 持续输出"
    assert lines[2].strip() == "action light_attack"


def test_format_tick_continuation_aligned():
    out = format_tick(_state(), "EXPLORE: 覆盖漫游", Action("move", {"direction": "forward"}))
    lines = out.splitlines()
    prefix_width = len("[12:00:01.123] ")
    assert lines[1].startswith(" " * prefix_width)
    assert lines[2].startswith(" " * prefix_width)
    assert lines[2].strip() == "action move direction=forward"


def test_format_tick_enemy_absent():
    state = _state()
    state.raw["enemy_hp_ratio"] = None
    state.raw["mp_visible"] = False  # 非战斗 MP 条隐藏 → 显示 -
    state.scene = "explore"
    lines = format_tick(state, "EXPLORE: 覆盖漫游", Action("turn", {"direction": "left", "degrees": 30})).splitlines()
    assert "enemy_hp=-" in lines[0]
    assert "mp=-" in lines[0]
    assert "scene=explore" in lines[0]
