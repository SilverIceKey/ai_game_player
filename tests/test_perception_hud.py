"""合成帧 HUD 感知测试：用 numpy 构造假血条/敌方血条喂给 WukongAdapter.perceive。"""
import numpy as np
import pytest

from games.wukong.adapter import WukongAdapter, WukongConfig

RED = (0, 0, 255)       # BGR → HSV(0, 255, 255)
GREEN = (0, 255, 0)     # BGR → HSV(60, 255, 255)
YELLOW = (0, 255, 255)  # BGR → HSV(30, 255, 255)
WHITE = (255, 255, 255)  # BGR → HSV(0, 0, 255)


@pytest.fixture(scope="module")
def adapter() -> WukongAdapter:
    return WukongAdapter(WukongConfig.load("configs/wukong.yaml"))


def _blank() -> np.ndarray:
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


def _fill_bar(frame: np.ndarray, rect: tuple[int, int, int, int], ratio: float, color) -> None:
    x, y, w, h = rect
    frame[y : y + h, x : x + int(round(w * ratio))] = color


def _fill_block(frame: np.ndarray, rect: tuple[int, int, int, int], color) -> None:
    x, y, w, h = rect
    frame[y : y + h, x : x + w] = color


def _hud_frame(adapter: WukongAdapter, hp=0.8, stamina=0.5, enemy=None, gourd=True, dead=False):
    frame = _blank()
    hud = adapter.config.hud
    _fill_bar(frame, hud.hp_bar.rect, hp, RED)
    _fill_bar(frame, hud.stamina_bar.rect, stamina, GREEN)
    if enemy is not None:
        _fill_bar(frame, hud.enemy_hp_bar.rect, enemy, RED)
    if gourd:
        _fill_block(frame, hud.gourd.rect, YELLOW)
    if dead:
        _fill_block(frame, hud.dead_indicator.rect, WHITE)
    return frame


def test_perceive_explore_scene(adapter):
    state = adapter.perceive(_hud_frame(adapter, hp=0.8, stamina=0.5))
    assert state.scene == "explore"
    assert state.raw["hp_ratio"] == pytest.approx(0.8, abs=0.05)
    assert state.raw["stamina_ratio"] == pytest.approx(0.5, abs=0.05)
    assert state.raw["gourd_available"] is True
    assert state.raw["enemy_hp_ratio"] is None
    assert state.raw["enemy_present"] is False
    assert state.raw["in_combat"] is False
    assert len(state.raw["pose"]) == 3
    assert set(state.raw["walkable"]) >= {"left", "center", "right", "suggestion"}


def test_perceive_combat_scene(adapter):
    state = adapter.perceive(_hud_frame(adapter, enemy=0.45))
    assert state.scene == "combat"
    assert state.raw["enemy_hp_ratio"] == pytest.approx(0.45, abs=0.05)
    assert state.raw["enemy_present"] is True
    assert state.raw["in_combat"] is True


def test_perceive_dead_scene(adapter):
    # 死亡提示优先级高于战斗：同时有敌方血条也应判定 dead
    state = adapter.perceive(_hud_frame(adapter, enemy=0.5, dead=True))
    assert state.scene == "dead"


def test_perceive_gourd_absent(adapter):
    state = adapter.perceive(_hud_frame(adapter, gourd=False))
    assert state.raw["gourd_available"] is False


def test_available_actions_by_scene(adapter):
    assert adapter.available_actions(type("S", (), {"scene": "dead"})()) == []
    combat = [a.name for a in adapter.available_actions(type("S", (), {"scene": "combat"})())]
    assert "light_attack" in combat and "heal" in combat
    explore = [a.name for a in adapter.available_actions(type("S", (), {"scene": "explore"})())]
    assert "move" in explore and "turn" in explore


def test_action_space(adapter):
    space = adapter.action_space()
    for name in ("move", "turn", "light_attack", "dodge", "heal", "lock_on"):
        assert name in space
