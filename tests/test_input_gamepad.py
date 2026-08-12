"""capture/input/gamepad.py 单元测试。

不触达真实 XInput DLL：解析逻辑用构造的假 state（结构体字节 / 原始字段）
直接测；Capture 线程用注入的假 poller 驱动。非 Windows 只验证
XInputPoller 构造的明确报错。
"""
from __future__ import annotations

import ctypes
import sys
import threading
import time

import pytest

from capture.action import SOURCE_HUMAN
from capture.input.gamepad import (
    BUTTON_BITS,
    XINPUT_STATE,
    XInputPoller,
    GamepadCapture,
    _action_changed,
    parse_gamepad_state,
    parse_state_bytes,
)
from capture.action import NormalizedAction


def _state_bytes(**overrides) -> bytes:
    """构造一份 XINPUT_STATE 原生字节（默认全零 = 无输入）。"""
    state = XINPUT_STATE()
    for field, value in overrides.items():
        setattr(state.Gamepad, field, value)
    return bytes(state)


# ---------- 原始状态解析（纯函数） ----------


def test_neutral_state():
    action = parse_gamepad_state(0, 0, 0, 0, 0, 0, 0)
    assert action.is_neutral()


def test_button_bit_mapping():
    action = parse_gamepad_state(BUTTON_BITS["a"], 0, 0, 0, 0, 0, 0)
    assert action.buttons == frozenset({"jump"})
    action = parse_gamepad_state(BUTTON_BITS["b"] | BUTTON_BITS["x"], 0, 0, 0, 0, 0, 0)
    assert action.buttons == frozenset({"dodge", "attack_light"})


def test_default_map_covers_required_buttons():
    cases = {
        "y": "interact",
        "left_shoulder": "block",
        "right_shoulder": "attack_heavy",
        "start": "wait",
        "back": "lock_target",
    }
    for bit_name, expected in cases.items():
        action = parse_gamepad_state(BUTTON_BITS[bit_name], 0, 0, 0, 0, 0, 0)
        assert action.buttons == frozenset({expected}), bit_name


def test_unmapped_buttons_ignored():
    action = parse_gamepad_state(
        BUTTON_BITS["dpad_up"] | BUTTON_BITS["left_thumb"], 0, 0, 0, 0, 0, 0
    )
    assert action.buttons == frozenset()


def test_stick_normalization_full_deflection():
    action = parse_gamepad_state(0, 0, 0, 32767, 32767, -32768, 0)
    assert action.move_x == pytest.approx(1.0)
    assert action.move_y == pytest.approx(1.0)  # LY 向上为正 = 前进
    assert action.camera_x == pytest.approx(-1.0)


def test_stick_deadzone():
    raw = int(32767 * 0.10)  # 死区 0.15 内
    action = parse_gamepad_state(0, 0, 0, raw, -raw, raw, -raw)
    assert action.move_x == 0.0 and action.move_y == 0.0
    assert action.camera_x == 0.0 and action.camera_y == 0.0


def test_stick_outside_deadzone_kept():
    raw = int(32767 * 0.5)
    action = parse_gamepad_state(0, 0, 0, raw, 0, 0, 0)
    assert action.move_x == pytest.approx(0.5, abs=1e-3)


def test_trigger_threshold_maps_to_buttons():
    # LT=200/255≈0.78 ≥ 0.5 → parry；RT=100/255≈0.39 < 0.5 → 不触发 heal
    action = parse_gamepad_state(0, 200, 100, 0, 0, 0, 0)
    assert action.buttons == frozenset({"parry"})
    action = parse_gamepad_state(0, 200, 255, 0, 0, 0, 0)
    assert action.buttons == frozenset({"parry", "heal"})


def test_custom_button_map_override():
    custom = {"a": "attack_light", "b": "dodge"}
    action = parse_gamepad_state(BUTTON_BITS["a"], 0, 0, 0, 0, 0, 0, button_map=custom)
    assert action.buttons == frozenset({"attack_light"})
    # 自定义映射之外的默认按钮不再生效
    action = parse_gamepad_state(BUTTON_BITS["start"], 0, 0, 0, 0, 0, 0, button_map=custom)
    assert action.buttons == frozenset()


# ---------- 假 state 字节解析 ----------


def test_parse_state_bytes_roundtrip():
    buf = _state_bytes(
        wButtons=BUTTON_BITS["a"] | BUTTON_BITS["right_shoulder"],
        bLeftTrigger=255,
        sThumbLX=32767,
        sThumbRY=int(32767 * 0.5),
    )
    action = parse_state_bytes(buf)
    assert action.buttons == frozenset({"jump", "attack_heavy", "parry"})
    assert action.move_x == pytest.approx(1.0)
    assert action.camera_y == pytest.approx(0.5, abs=1e-3)


def test_state_struct_layout_is_16_bytes():
    # XINPUT_STATE 原生布局：DWORD + 12 字节 GAMEPAD；布局漂移会破坏 DLL 交互
    assert ctypes.sizeof(XINPUT_STATE) == 16


# ---------- epsilon 变化检测 ----------


def test_action_changed_epsilon():
    base = NormalizedAction(move_x=0.5)
    assert _action_changed(None, base, 0.02) is True
    assert _action_changed(base, NormalizedAction(move_x=0.51), 0.02) is False
    assert _action_changed(base, NormalizedAction(move_x=0.53), 0.02) is True
    assert _action_changed(base, NormalizedAction(move_x=0.5, buttons={"jump"}), 0.02) is True


# ---------- 假 poller 注入的 Capture 线程 ----------


class _FakePoller:
    """按脚本返回手柄状态；脚本播完后保持最后一帧。"""

    def __init__(self, script: list[tuple | None]):
        self._script = list(script)
        self._lock = threading.Lock()

    def poll(self):
        with self._lock:
            if len(self._script) > 1:
                return self._script.pop(0)
            return self._script[0] if self._script else None


def _raw(buttons=0, lt=0, rt=0, lx=0, ly=0, rx=0, ry=0):
    return (buttons, lt, rt, lx, ly, rx, ry)


def _drain(capture: GamepadCapture, duration: float = 0.15):
    time.sleep(duration)
    capture.stop()
    records = []
    while True:
        record = capture.poll(timeout=0.01)
        if record is None:
            return records
        records.append(record)


def test_capture_emits_on_state_change_only():
    neutral = _raw()
    pressed = _raw(buttons=BUTTON_BITS["a"])
    poller = _FakePoller([neutral, neutral, neutral, pressed, pressed])
    capture = GamepadCapture(poller=poller, poll_hz=500.0)
    capture.start()
    records = _drain(capture)

    # 首帧建立基线快照 + 按下 A 一次变化；重复状态不重复发
    assert len(records) == 2
    assert records[0].action.is_neutral()
    assert records[1].action.pressed("jump")
    assert all(r.source == SOURCE_HUMAN for r in records)
    assert all(r.timestamp_us > 0 for r in records)
    assert records[1].timestamp_us >= records[0].timestamp_us


def test_capture_emits_axis_change_beyond_epsilon():
    small = _raw(lx=int(32767 * 0.20))
    big = _raw(lx=int(32767 * 0.80))
    poller = _FakePoller([_raw(), small, big])
    capture = GamepadCapture(poller=poller, poll_hz=500.0)
    capture.start()
    records = _drain(capture)
    assert [round(r.action.move_x, 2) for r in records] == [0.0, 0.2, 0.8]


def test_capture_disconnected_poller_emits_nothing():
    capture = GamepadCapture(poller=_FakePoller([None]), poll_hz=500.0)
    capture.start()
    assert _drain(capture) == []


def test_capture_start_stop_idempotent():
    capture = GamepadCapture(poller=_FakePoller([_raw()]), poll_hz=500.0)
    capture.start()
    capture.start()  # 幂等
    capture.stop()
    capture.stop()  # 幂等
    assert capture.poll(timeout=0.05) is not None  # 基线快照仍在队列中


def test_capture_poller_error_degrades_with_log(caplog):
    class _BoomPoller:
        def poll(self):
            raise RuntimeError("XInputGetState 失败: 错误码 1")

    capture = GamepadCapture(poller=_BoomPoller(), poll_hz=500.0)
    capture.start()
    with caplog.at_level("ERROR"):
        records = _drain(capture)
    assert records == []  # 降级为无事件，线程不死
    assert "手柄轮询失败" in caplog.text


# ---------- 平台门槛 ----------


def test_xinput_poller_raises_on_non_windows():
    if sys.platform == "win32":
        pytest.skip("仅非 Windows 平台验证报错路径")
    with pytest.raises(RuntimeError, match="仅支持 Windows"):
        XInputPoller()


def test_capture_constructor_raises_on_non_windows_without_poller():
    if sys.platform == "win32":
        pytest.skip("仅非 Windows 平台验证报错路径")
    with pytest.raises(RuntimeError, match="仅支持 Windows"):
        GamepadCapture()
