"""capture/input/keyboard_mouse.py 单元测试。

全部测试不 import pynput：纯逻辑打 KeyMouseMapper；pynput 键对象转换用
鸭子类型的假对象；Capture 层用假键对象直接驱动内部回调验证 queue 集成。
"""
from __future__ import annotations

import math
import sys
from types import ModuleType, SimpleNamespace

import pytest

from capture.action import SOURCE_HUMAN
from capture.input.keyboard_mouse import (
    KeyboardMouseCapture,
    KeyMouseMapper,
    _physical_win32_event,
    _pynput_key_to_str,
    _pynput_mouse_button_to_str,
    build_reverse_keymap,
    canonical_key,
)

# 模拟一份 config keys 段（动作 → 键位）
KEYS = {
    "move_forward": "w",
    "move_back": "s",
    "move_left": "a",
    "move_right": "d",
    "dodge": "space",
    "attack_light": "mouse_left",
    "attack_heavy": "mouse_right",
    "heal": "F",
}


@pytest.fixture
def mapper() -> KeyMouseMapper:
    return KeyMouseMapper(build_reverse_keymap(KEYS), norm_pixels=50.0)


# ---------- 键名规范化与反向映射 ----------


def test_canonical_key():
    assert canonical_key("  W ") == "w"
    assert canonical_key("F12") == "f12"


def test_build_reverse_keymap():
    reverse = build_reverse_keymap(KEYS)
    assert reverse["w"] == "move_forward"
    assert reverse["f"] == "heal"  # 大小写不敏感
    assert reverse["mouse_left"] == "attack_light"


def test_build_reverse_keymap_conflict_raises():
    with pytest.raises(ValueError, match="键位冲突"):
        build_reverse_keymap({"dodge": "space", "jump": "SPACE"})


# ---------- move 向量合成 ----------


def test_single_move_key(mapper):
    action = mapper.on_key("w", True)
    assert action is not None
    assert action.move_y == 1.0 and action.move_x == 0.0


def test_move_key_release_returns_to_neutral(mapper):
    mapper.on_key("w", True)
    action = mapper.on_key("w", False)
    assert action is not None and action.is_neutral()


def test_diagonal_move_normalized_to_unit_length(mapper):
    mapper.on_key("w", True)
    action = mapper.on_key("d", True)
    assert action is not None
    assert math.isclose(action.move_x, math.sqrt(0.5), abs_tol=1e-9)
    assert math.isclose(action.move_y, math.sqrt(0.5), abs_tol=1e-9)


def test_opposite_move_keys_cancel(mapper):
    mapper.on_key("w", True)
    action = mapper.on_key("s", True)
    assert action is not None
    assert action.move_y == 0.0


def test_partial_release_keeps_remaining_direction(mapper):
    mapper.on_key("w", True)
    mapper.on_key("d", True)
    action = mapper.on_key("d", False)
    assert action is not None
    assert action.move_y == 1.0 and action.move_x == 0.0


# ---------- 按钮映射 ----------


def test_button_key_press_and_release(mapper):
    action = mapper.on_key("space", True)
    assert action is not None and action.pressed("dodge")
    action = mapper.on_key("space", False)
    assert action is not None and not action.buttons


def test_mouse_button_maps_to_action(mapper):
    action = mapper.on_mouse_button("mouse_left", True)
    assert action is not None and action.pressed("attack_light")


def test_unmapped_key_ignored(mapper):
    assert mapper.on_key("q", True) is None
    assert mapper.on_mouse_button("mouse_middle", True) is None


def test_snapshot_combines_move_and_buttons(mapper):
    mapper.on_key("w", True)
    action = mapper.on_key("space", True)
    assert action is not None
    assert action.move_y == 1.0 and action.pressed("dodge")


# ---------- 鼠标 delta 归一化与合并 ----------


def test_mouse_move_accumulates_and_flushes(mapper):
    mapper.on_mouse_move(25.0, -25.0)  # 右移 25px、上移 25px
    mapper.on_mouse_move(0.0, 0.0)
    action = mapper.flush_camera()
    assert action is not None
    assert action.camera_x == pytest.approx(0.5)  # 25 / norm_pixels(50)
    assert action.camera_y == pytest.approx(0.5)  # 上移为正（dy 取反）


def test_mouse_delta_clamped_to_unit(mapper):
    mapper.on_mouse_move(500.0, -500.0)
    action = mapper.flush_camera()
    assert action is not None
    assert action.camera_x == 1.0 and action.camera_y == 1.0


def test_flush_without_movement_returns_none(mapper):
    assert mapper.flush_camera() is None


def test_flush_carries_current_move_and_buttons(mapper):
    mapper.on_key("w", True)
    mapper.on_key("space", True)
    mapper.on_mouse_move(10.0, 0.0)
    action = mapper.flush_camera()
    assert action is not None
    assert action.move_y == 1.0 and action.pressed("dodge") and action.camera_x > 0


def test_flush_resets_accumulator(mapper):
    mapper.on_mouse_move(10.0, 0.0)
    mapper.flush_camera()
    assert mapper.flush_camera() is None


def test_invalid_norm_pixels_raises():
    with pytest.raises(ValueError):
        KeyMouseMapper({}, norm_pixels=0)


# ---------- pynput 键对象转换（鸭子类型假对象，不 import pynput） ----------


def test_pynput_keycode_char():
    assert _pynput_key_to_str(SimpleNamespace(char="W")) == "w"


def test_pynput_special_key_name():
    assert _pynput_key_to_str(SimpleNamespace(char=None, name="f12")) == "f12"


def test_pynput_unknown_key_returns_none():
    assert _pynput_key_to_str(object()) is None


def test_pynput_mouse_button_names():
    assert _pynput_mouse_button_to_str(SimpleNamespace(name="left")) == "mouse_left"
    assert _pynput_mouse_button_to_str(SimpleNamespace(name="right")) == "mouse_right"
    assert _pynput_mouse_button_to_str(SimpleNamespace(name="x1")) is None


def test_win32_injected_events_are_filtered_but_physical_events_pass():
    assert not _physical_win32_event(SimpleNamespace(flags=0x10), 0x10)
    assert not _physical_win32_event(SimpleNamespace(flags=0x01), 0x01)
    assert _physical_win32_event(SimpleNamespace(flags=0), 0x10)
    assert _physical_win32_event(SimpleNamespace(flags=0), 0x01)


def test_capture_installs_win32_source_filters(monkeypatch):
    listeners = []

    class Listener:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            listeners.append(self)

        def start(self):
            pass

        def stop(self):
            pass

    pynput = ModuleType("pynput")
    pynput.keyboard = SimpleNamespace(Listener=Listener)
    pynput.mouse = SimpleNamespace(Listener=Listener)
    monkeypatch.setitem(sys.modules, "pynput", pynput)
    monkeypatch.setattr("capture.input.keyboard_mouse.sys.platform", "win32")

    capture = KeyboardMouseCapture(build_reverse_keymap(KEYS))
    capture.start()
    keyboard_filter = listeners[0].kwargs["win32_event_filter"]
    mouse_filter = listeners[1].kwargs["win32_event_filter"]
    assert not keyboard_filter(0, SimpleNamespace(flags=0x10))
    assert keyboard_filter(0, SimpleNamespace(flags=0))
    assert not mouse_filter(0, SimpleNamespace(flags=0x01))
    assert mouse_filter(0, SimpleNamespace(flags=0))
    capture.stop()


# ---------- Capture 层：queue 集成与依赖缺失降级 ----------


def test_capture_start_without_pynput_raises_clear_error():
    try:
        import pynput  # noqa: F401
        pytest.skip("环境已安装 pynput，跳过依赖缺失路径")
    except ImportError:
        pass
    capture = KeyboardMouseCapture(build_reverse_keymap(KEYS))
    with pytest.raises(RuntimeError, match="pynput"):
        capture.start()


def test_capture_emit_and_poll_roundtrip():
    """不 start 监听器，直接驱动内部事件回调验证事件入队 + poll 取件。"""
    capture = KeyboardMouseCapture(build_reverse_keymap(KEYS))
    capture._on_key_event(SimpleNamespace(char="w"), True)
    record = capture.poll(timeout=0.1)
    assert record is not None
    assert record.source == SOURCE_HUMAN
    assert record.timestamp_us > 0
    assert record.action.move_y == 1.0
    assert capture.poll(timeout=0.01) is None  # 队列空时超时返回 None


def test_capture_mouse_move_delta_uses_position_diff():
    """pynput on_move 给绝对坐标，Capture 负责差分；首个事件只建基准。"""
    capture = KeyboardMouseCapture(build_reverse_keymap(KEYS), norm_pixels=50.0)
    capture._on_move(100.0, 100.0)  # 基准点
    capture._on_move(120.0, 90.0)  # dx=+20, dy=-10
    with capture._lock:
        action = capture._mapper.flush_camera()
    assert action is not None
    assert action.camera_x == pytest.approx(0.4)
    assert action.camera_y == pytest.approx(0.2)
