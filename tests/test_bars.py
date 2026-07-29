"""动态小怪血条检测（bars.py）与 adapter 集成测试：Boss 优先 / hp_visible。"""
import numpy as np
import pytest

from core.perception.bars import BarSearchSpec, detect_bar
from core.perception.regions import ColorRange
from games.wukong.adapter import WukongAdapter, WukongConfig

RED = (0, 0, 255)
TRACK = (40, 40, 40)  # 暗色背景槽（HSV V=40, S=0，落入 track_hsv 范围）
SPEC = BarSearchSpec(
    rect=(560, 200, 800, 500),
    color=ColorRange((0, 120, 80), (10, 255, 255)),
    track_color=ColorRange((0, 0, 20), (179, 80, 90)),
    min_length=40,
    min_aspect=3.0,
    min_fill=0.5,
    column_fill=0.3,
)


def _blank() -> np.ndarray:
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


def _draw_bar(frame, x, y, w, h, ratio, color=RED):
    """画一条真实结构的小怪血条：全长背景槽 + 前方填充。"""
    frame[y : y + h, x : x + w] = TRACK
    frame[y : y + h, x : x + int(round(w * ratio))] = color


def test_detect_floating_bar():
    frame = _blank()
    _draw_bar(frame, 700, 300, 120, 8, 0.6)
    detected = detect_bar(frame, SPEC)
    assert detected is not None
    assert detected.ratio == pytest.approx(0.6, abs=0.05)
    bx, by, bw, bh = detected.box
    assert abs(bx - 700) <= 1 and abs(by - 300) <= 1
    assert 110 <= bw <= 120


def test_detect_no_bar_returns_none():
    assert detect_bar(_blank(), SPEC) is None


def test_detect_picks_nearest_to_center():
    frame = _blank()
    _draw_bar(frame, 600, 250, 100, 8, 0.8)   # 中心距 ≈ 422
    _draw_bar(frame, 1200, 600, 100, 8, 0.4)  # 中心距 ≈ 297（更近）
    detected = detect_bar(frame, SPEC)
    assert detected is not None
    assert abs(detected.box[0] - 1200) <= 1
    assert detected.ratio == pytest.approx(0.4, abs=0.05)


def test_shape_filters():
    # 方块：长宽比不足 → 拒绝
    frame = _blank()
    frame[300:330, 700:730] = TRACK
    assert detect_bar(frame, SPEC) is None
    # 短条：长度不足 → 拒绝
    frame = _blank()
    frame[300:306, 700:720] = TRACK
    assert detect_bar(frame, SPEC) is None
    # 搜索区域外的血条 → 不检出
    frame = _blank()
    _draw_bar(frame, 100, 100, 120, 8, 0.9)
    assert detect_bar(frame, SPEC) is None


# ---------- adapter 集成 ----------

@pytest.fixture(scope="module")
def adapter() -> WukongAdapter:
    return WukongAdapter(WukongConfig.load("configs/wukong.yaml"))


def test_adapter_dynamic_enemy_bar(adapter):
    """浮头血条出现在搜索区域任意位置 → 接敌，source=dynamic。"""
    frame = _blank()
    _draw_bar(frame, 900, 350, 120, 8, 0.6)
    state = adapter.perceive(frame)
    assert state.scene == "combat"
    assert state.raw["enemy_hp_ratio"] == pytest.approx(0.6, abs=0.05)
    assert state.raw["enemy_hp_source"] == "dynamic"
    assert state.raw["enemy_bar_box"] is not None
    # 自身血条未画（非战斗隐藏）→ hp_visible=false 且 hp 按 1.0
    assert state.raw["hp_visible"] is False
    assert state.raw["hp_ratio"] == 1.0


def test_adapter_boss_bar_priority(adapter):
    """Boss 固定条与动态条同时出现时 Boss 优先。"""
    hud = adapter.config.hud
    frame = _blank()
    x, y, w, h = hud.enemy_hp_bar.rect
    _draw_bar(frame, x, y, w, h, 0.5)
    _draw_bar(frame, 900, 350, 120, 8, 0.9)  # 动态条（搜索区域内）
    state = adapter.perceive(frame)
    assert state.raw["enemy_hp_source"] == "boss"
    assert state.raw["enemy_hp_ratio"] == pytest.approx(0.5, abs=0.05)


def test_adapter_no_enemy_no_bar(adapter):
    state = adapter.perceive(_blank())
    assert state.scene == "explore"
    assert state.raw["enemy_hp_ratio"] is None
    assert state.raw["enemy_hp_source"] is None
    assert state.raw["enemy_present"] is False
    assert state.raw["hp_visible"] is False
    assert state.raw["hp_ratio"] == 1.0


def test_adapter_hp_visible_when_bar_drawn(adapter):
    frame = _blank()
    x, y, w, h = adapter.config.hud.hp_bar.rect
    _draw_bar(frame, x, y, w, h, 0.75)
    state = adapter.perceive(frame)
    assert state.raw["hp_visible"] is True
    assert state.raw["hp_ratio"] == pytest.approx(0.75, abs=0.05)
