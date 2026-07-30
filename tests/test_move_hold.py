"""hold 移动模式测试：持续按住、方向切换、停止与会话结束释放。"""
import time

import pytest

from core.contracts import Action
from core.control.directinput import ControlParams, DirectInputController


class _FakePdi:
    def __init__(self):
        self.events = []  # ("down"|"up", key)

    def keyDown(self, key):
        self.events.append(("down", key))

    def keyUp(self, key):
        self.events.append(("up", key))


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _s: None)


def _controller(move_mode: str = "hold") -> tuple[DirectInputController, _FakePdi]:
    controller = DirectInputController(
        {"move_forward": "w", "move_back": "s"}, ControlParams(move_mode=move_mode)
    )
    fake = _FakePdi()
    controller._pdi = fake
    return controller, fake


def _forward(controller, times=1):
    for _ in range(times):
        controller.execute(Action("move", {"direction": "forward"}))


def test_hold_mode_presses_once_for_consecutive_moves():
    controller, fake = _controller()
    _forward(controller, 3)
    assert fake.events == [("down", "w")]  # 持续按住，不重复点按


def test_hold_mode_switches_direction():
    controller, fake = _controller()
    _forward(controller)
    controller.execute(Action("move", {"direction": "back"}))
    assert fake.events == [("down", "w"), ("up", "w"), ("down", "s")]


def test_idle_releases_held_keys():
    controller, fake = _controller()
    _forward(controller)
    controller.execute(Action("idle"))
    assert fake.events == [("down", "w"), ("up", "w")]
    # idle 后可以重新按住
    _forward(controller)
    assert fake.events[-1] == ("down", "w")


def test_release_all_on_session_end():
    controller, fake = _controller()
    _forward(controller, 2)
    controller.release_all()
    assert ("up", "w") in fake.events
    assert controller._held_move == set()


def test_tap_mode_legacy_behavior():
    controller, fake = _controller("tap")
    _forward(controller, 2)
    assert fake.events == [("down", "w"), ("up", "w"), ("down", "w"), ("up", "w")]
