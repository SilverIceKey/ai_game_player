"""平滑转向与探索迟滞/航向承诺测试。"""
import time

import pytest

from core.contracts import Action
from core.control.directinput import ControlParams, DirectInputController
from core.decision.navigation import CoverageExplorer, ExplorationParams
from core.navigation.grid_map import OccupancyGrid
from core.perception.walkable import WalkableResult


# ---------- 平滑转向 ----------

class _FakePdi:
    def __init__(self):
        self.moves = []

    def moveRel(self, dx, dy, relative=True):
        self.moves.append((dx, dy))


def _controller_with_fake(params: ControlParams) -> tuple[DirectInputController, _FakePdi]:
    controller = DirectInputController({}, params)
    fake = _FakePdi()
    controller._pdi = fake  # 注入伪后端，跳过 pydirectinput 延迟导入
    return controller, fake


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _s: None)


def test_turn_is_split_into_linear_steps():
    params = ControlParams(
        pixels_per_degree=12.0, turn_degrees_per_second=180.0,
        turn_step_interval=0.02, action_pause=0.0,
    )
    controller, fake = _controller_with_fake(params)
    result = controller.execute(Action("turn", {"degrees": 30.0, "direction": "right"}))

    assert result.success
    # 30° × 12px/° = 360px；0.167s / 0.02s ≈ 8 步——多步而非一次瞬移
    assert len(fake.moves) > 1
    assert sum(dx for dx, _ in fake.moves) == pytest.approx(360, abs=1)
    assert all(dy == 0 for _, dy in fake.moves)
    # 线性：各步像素量基本一致（残差累积允许末步差 1px）
    amounts = [dx for dx, _ in fake.moves]
    assert max(amounts) - min(amounts) <= 1


def test_turn_left_is_negative():
    controller, fake = _controller_with_fake(
        ControlParams(action_pause=0.0, turn_step_interval=0.02)
    )
    controller.execute(Action("turn", {"degrees": 30.0, "direction": "left"}))
    assert all(dx < 0 for dx, _ in fake.moves)


def test_tiny_turn_still_delivers_pixels():
    # 极小转角（不足一步）也要把像素发出去，不能吞掉
    controller, fake = _controller_with_fake(
        ControlParams(pixels_per_degree=1.5, action_pause=0.0)
    )
    controller.execute(Action("turn", {"degrees": 2.0, "direction": "right"}))
    assert sum(dx for dx, _ in fake.moves) == 3


# ---------- 探索迟滞与航向承诺 ----------

def _walkable(left: float, center: float, right: float) -> WalkableResult:
    return WalkableResult(left=left, center=center, right=right, suggestion="straight")


def _explorer() -> CoverageExplorer:
    return CoverageExplorer(OccupancyGrid(60.0, 0.5), ExplorationParams())


def test_hysteresis_blocks_oscillation():
    """评分噪声导致的左右交替不应引发转向摇摆（实机反馈"转半天又转回来"）。"""
    explorer = _explorer()
    pose = (0.0, 0.0, 0.0)
    # 正前方明显最优，左右在噪声范围内交替略高——但不超过迟滞阈值
    actions = [
        explorer.decide(pose, _walkable(0.7, 0.9, 0.6)).name,
        explorer.decide(pose, _walkable(0.6, 0.9, 0.7)).name,
        explorer.decide(pose, _walkable(0.7, 0.9, 0.6)).name,
    ]
    assert actions == ["lock_on", "move", "move"] or actions == ["move", "move", "move"]
    # 关键断言：没有任何一次选择转向
    assert "turn" not in actions


def test_clear_side_advantage_turns_and_commits():
    """侧向明显更优时转向，且承诺期内不因轻微噪声改向。"""
    explorer = _explorer()
    pose = (0.0, 0.0, 0.0)
    first = explorer.decide(pose, _walkable(0.95, 0.2, 0.1))
    assert first.name == "turn" and first.params["direction"] == "left"
    # 承诺期内噪声翻转左右评分：仍沿承诺方向（left）转，不摇摆
    second = explorer.decide(pose, _walkable(0.3, 0.25, 0.4))
    assert second.name == "turn" and second.params["direction"] == "left"
