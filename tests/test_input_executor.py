"""runtime/input_executor.py 单元测试（spec §9/§10：NormalizedAction → 键鼠）。

pydirectinput 用假 backend 注入，不触达真实键鼠；sleep 也注入，避免真实等待。
"""
from __future__ import annotations

from capture.action import NormalizedAction
from config import ExecutorConfig
from runtime.input_executor import KeyboardMouseExecutor

KEYMAP = {
    "move_forward": "w",
    "move_back": "s",
    "move_left": "a",
    "move_right": "d",
    "attack_light": "mouse_left",
    "dodge": "space",
    "heal": "r",
}

PARAMS = ExecutorConfig(pixels_per_unit=400.0, action_pause=0.0, move_deadzone=0.15)


class FakeBackend:
    """记录全部调用的假 pydirectinput。"""

    def __init__(self):
        self.calls: list[tuple] = []

    def keyDown(self, key):
        self.calls.append(("keyDown", key))

    def keyUp(self, key):
        self.calls.append(("keyUp", key))

    def mouseDown(self, button=None):
        self.calls.append(("mouseDown", button))

    def mouseUp(self, button=None):
        self.calls.append(("mouseUp", button))

    def moveRel(self, x, y, relative=True):
        self.calls.append(("moveRel", x, y, relative))


def _executor(backend: FakeBackend, params: ExecutorConfig = PARAMS, keymap=None):
    return KeyboardMouseExecutor(keymap or KEYMAP, params, backend=backend, sleep=lambda s: None)


class TestMoveDifferential:
    def test_hold_new_direction_and_release_vanished(self):
        backend = FakeBackend()
        ex = _executor(backend)
        detail = ex.execute(NormalizedAction(move_y=1.0))
        assert ("keyDown", "w") in backend.calls
        assert "hold:w" in detail

        # 同方向重复执行是 no-op
        backend.calls.clear()
        ex.execute(NormalizedAction(move_y=1.0))
        assert backend.calls == []

        # 方向消失 → keyUp；新方向 → keyDown
        backend.calls.clear()
        detail = ex.execute(NormalizedAction(move_x=-1.0))
        assert ("keyUp", "w") in backend.calls
        assert ("keyDown", "a") in backend.calls
        assert "release:w" in detail and "hold:a" in detail

    def test_deadzone_no_key(self):
        backend = FakeBackend()
        ex = _executor(backend)
        assert ex.execute(NormalizedAction(move_y=0.1)) == "idle"
        assert backend.calls == []

    def test_diagonal_holds_two_keys(self):
        backend = FakeBackend()
        ex = _executor(backend)
        ex.execute(NormalizedAction(move_x=1.0, move_y=1.0))
        assert ("keyDown", "w") in backend.calls
        assert ("keyDown", "d") in backend.calls


class TestButtonDifferential:
    def test_press_and_release_diff(self):
        backend = FakeBackend()
        ex = _executor(backend)
        ex.execute(NormalizedAction(buttons=frozenset({"dodge"})))
        assert ("keyDown", "space") in backend.calls

        backend.calls.clear()
        ex.execute(NormalizedAction(buttons=frozenset({"dodge"})))
        assert backend.calls == []  # 持续按住无重复敲击

        backend.calls.clear()
        ex.execute(NormalizedAction.neutral())
        assert ("keyUp", "space") in backend.calls

    def test_mouse_button_uses_mouse_down_up(self):
        backend = FakeBackend()
        ex = _executor(backend)
        ex.execute(NormalizedAction(buttons=frozenset({"attack_light"})))
        assert ("mouseDown", "left") in backend.calls
        backend.calls.clear()
        ex.execute(NormalizedAction.neutral())
        assert ("mouseUp", "left") in backend.calls

    def test_unknown_action_reports_error_without_crash(self):
        backend = FakeBackend()
        ex = _executor(backend)
        detail = ex.execute(NormalizedAction(buttons=frozenset({"skill_1"})))  # keymap 未配置
        assert "error" in detail
        assert "skill_1" in detail
        assert backend.calls == []  # 未配置的键位不产生任何输入


class TestCameraPixels:
    def test_camera_to_pixels(self):
        backend = FakeBackend()
        ex = _executor(backend)
        detail = ex.execute(NormalizedAction(camera_x=0.5, camera_y=0.25))
        # 0.5*400=200 像素右转；camera_y 上为正 → 鼠标上移 = 屏幕 y 负方向：-0.25*400=-100
        assert ("moveRel", 200, -100, True) in backend.calls
        assert "camera:dx=200,dy=-100" in detail

    def test_camera_below_rounding_threshold_no_move(self):
        backend = FakeBackend()
        ex = _executor(backend)
        assert ex.execute(NormalizedAction(camera_x=0.001)) == "idle"
        assert backend.calls == []


class TestReleaseAll:
    def test_releases_everything_held(self):
        backend = FakeBackend()
        ex = _executor(backend)
        ex.execute(NormalizedAction(move_y=1.0, buttons=frozenset({"attack_light", "dodge"})))
        backend.calls.clear()
        ex.release_all()
        assert ("keyUp", "w") in backend.calls
        assert ("mouseUp", "left") in backend.calls
        assert ("keyUp", "space") in backend.calls
        # 全部松开后，再次执行同一动作需要重新按下
        backend.calls.clear()
        ex.execute(NormalizedAction(move_y=1.0))
        assert ("keyDown", "w") in backend.calls

    def test_release_all_failure_does_not_raise(self):
        class FlakyBackend(FakeBackend):
            def keyUp(self, key):
                raise RuntimeError("device gone")

        ex = _executor(FlakyBackend())
        ex.execute(NormalizedAction(move_y=1.0, buttons=frozenset({"dodge"})))
        ex.release_all()  # 单个释放失败不抛出（安全兜底尽力而为）


class TestBackendUnavailable:
    def test_execute_returns_error_detail_when_pydirectinput_missing(self):
        ex = KeyboardMouseExecutor(KEYMAP, PARAMS, sleep=lambda s: None)
        ex._pdi_failed = True  # 模拟 Linux 开发机 import 失败（不触达真实 import）
        detail = ex.execute(NormalizedAction(move_y=1.0))
        assert "error" in detail
        assert "pydirectinput" in detail

    def test_release_all_noop_without_backend(self):
        ex = KeyboardMouseExecutor(KEYMAP, PARAMS, sleep=lambda s: None)
        ex._pdi_failed = True
        ex.release_all()  # 不抛异常


class TestActionPause:
    def test_pause_called_after_execute(self):
        slept: list[float] = []
        ex = KeyboardMouseExecutor(
            KEYMAP, ExecutorConfig(action_pause=0.02), backend=FakeBackend(), sleep=slept.append
        )
        ex.execute(NormalizedAction.neutral())
        assert slept == [0.02]
