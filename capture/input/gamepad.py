"""手柄输入采集：XInput via ctypes 轮询 → NormalizedAction 事件（spec §10/§11）。

设计（重构计划 §4.3）：采集用 XInput ctypes 轮询，零新第三方依赖；
所有 win32 调用隔离在 XInputPoller 内并延迟到构造时触达，
非 Windows 构造时抛明确 RuntimeError；Linux 开发机通过注入假 poller /
假 state 字节跑全部单元测试。

行为要点（spec §10：手柄数据优先使用连续轴）：
- 左摇杆 LX/LY → move_x/move_y（前为正：LY 向上为正，直接对应 move_y）
- 右摇杆 RX/RY → camera_x/camera_y（右转/上抬为正）
- 摇杆值除 32767 归一化到 [-1,1]，死区 0.15 内归零
- 扳机 LT/RT 除 255 归一化，超过阈值（默认 0.5）视为按钮按下
- wButtons 位掩码 + 扳机 → NormalizedAction buttons（DEFAULT_GAMEPAD_MAP，
  构造时可覆盖）
- 轮询线程默认 120Hz；状态变化超过 epsilon 才发 ActionRecord(source=human)，
  时间戳 = 收到该帧状态的 now_us()（spec §11）
"""
from __future__ import annotations

import ctypes
import logging
import sys
import threading

from capture.action import SOURCE_HUMAN, ActionRecord, NormalizedAction
from capture.clock import now_us
from capture.input.base import QueuedInputCapture

# XInput wButtons 位掩码（XINPUT_GAMEPAD 标准定义）
BUTTON_BITS: dict[str, int] = {
    "dpad_up": 0x0001,
    "dpad_down": 0x0002,
    "dpad_left": 0x0004,
    "dpad_right": 0x0008,
    "start": 0x0010,
    "back": 0x0020,
    "left_thumb": 0x0040,
    "right_thumb": 0x0080,
    "left_shoulder": 0x0100,
    "right_shoulder": 0x0200,
    "a": 0x1000,
    "b": 0x2000,
    "x": 0x4000,
    "y": 0x8000,
}

# 默认手柄键位 → 动作映射（§9 动作名；左摇杆=move、右摇杆=camera 为固定轴不占按钮）
DEFAULT_GAMEPAD_MAP: dict[str, str] = {
    "a": "jump",
    "b": "dodge",
    "x": "attack_light",
    "y": "interact",
    "left_shoulder": "block",
    "right_shoulder": "attack_heavy",
    "left_trigger": "parry",
    "right_trigger": "heal",
    "start": "wait",
    "back": "lock_target",
}

_TRIGGER_MAX = 255.0
_STICK_MAX = 32767.0
_ERROR_DEVICE_NOT_CONNECTED = 1167


class XINPUT_GAMEPAD(ctypes.Structure):
    """XINPUT_GAMEPAD 原生布局（12 字节）；ctypes 为 stdlib，定义本身跨平台安全。"""

    _fields_ = [
        ("wButtons", ctypes.c_uint16),
        ("bLeftTrigger", ctypes.c_uint8),
        ("bRightTrigger", ctypes.c_uint8),
        ("sThumbLX", ctypes.c_int16),
        ("sThumbLY", ctypes.c_int16),
        ("sThumbRX", ctypes.c_int16),
        ("sThumbRY", ctypes.c_int16),
    ]


class XINPUT_STATE(ctypes.Structure):
    """XINPUT_STATE 原生布局（16 字节）。"""

    _fields_ = [
        ("dwPacketNumber", ctypes.c_uint32),
        ("Gamepad", XINPUT_GAMEPAD),
    ]


def _stick_axis(raw: int, deadzone: float) -> float:
    """摇杆原始值 → [-1,1] 归一化；死区内归零。"""
    value = max(-1.0, min(1.0, raw / _STICK_MAX))
    if abs(value) < deadzone:
        return 0.0
    return value


def parse_gamepad_state(
    w_buttons: int,
    left_trigger: int,
    right_trigger: int,
    thumb_lx: int,
    thumb_ly: int,
    thumb_rx: int,
    thumb_ry: int,
    deadzone: float = 0.15,
    trigger_threshold: float = 0.5,
    button_map: dict[str, str] | None = None,
) -> NormalizedAction:
    """一份手柄原始状态 → NormalizedAction。纯函数，供单元测试直接覆盖。

    button_map 键名：BUTTON_BITS 的按钮名，加上 "left_trigger"/"right_trigger"
    两个模拟扳机（归一化超过 trigger_threshold 视为按下）。
    """
    mapping = DEFAULT_GAMEPAD_MAP if button_map is None else button_map

    buttons: set[str] = set()
    for bit_name, mask in BUTTON_BITS.items():
        action = mapping.get(bit_name)
        if action is not None and (w_buttons & mask):
            buttons.add(action)
    lt_action = mapping.get("left_trigger")
    if lt_action is not None and left_trigger / _TRIGGER_MAX >= trigger_threshold:
        buttons.add(lt_action)
    rt_action = mapping.get("right_trigger")
    if rt_action is not None and right_trigger / _TRIGGER_MAX >= trigger_threshold:
        buttons.add(rt_action)

    return NormalizedAction(
        move_x=_stick_axis(thumb_lx, deadzone),
        move_y=_stick_axis(thumb_ly, deadzone),  # XInput LY 向上为正 = 前进为正
        camera_x=_stick_axis(thumb_rx, deadzone),
        camera_y=_stick_axis(thumb_ry, deadzone),
        buttons=frozenset(buttons),
    )


def parse_state_bytes(
    buf: bytes,
    deadzone: float = 0.15,
    trigger_threshold: float = 0.5,
    button_map: dict[str, str] | None = None,
) -> NormalizedAction:
    """从 XINPUT_STATE 原生字节解析（测试用假 state 字节的入口）。"""
    state = XINPUT_STATE.from_buffer_copy(buf)
    g = state.Gamepad
    return parse_gamepad_state(
        g.wButtons, g.bLeftTrigger, g.bRightTrigger,
        g.sThumbLX, g.sThumbLY, g.sThumbRX, g.sThumbRY,
        deadzone=deadzone, trigger_threshold=trigger_threshold, button_map=button_map,
    )


def _action_changed(a: NormalizedAction | None, b: NormalizedAction, epsilon: float) -> bool:
    """状态是否发生超过 epsilon 的变化（按钮集合不同也算变化）。"""
    if a is None:
        return True
    if a.buttons != b.buttons:
        return True
    return (
        abs(a.move_x - b.move_x) > epsilon
        or abs(a.move_y - b.move_y) > epsilon
        or abs(a.camera_x - b.camera_x) > epsilon
        or abs(a.camera_y - b.camera_y) > epsilon
    )


class XInputPoller:
    """XInput DLL 轮询器：唯一触达 win32 的类（构造时加载 DLL）。

    优先 xinput1_4.dll（Win8+），失败回退 xinput9_1_0.dll（兼容旧系统）。
    非 Windows 或两个 DLL 都加载失败时抛明确 RuntimeError。
    """

    def __init__(self, user_index: int = 0):
        if sys.platform != "win32":
            raise RuntimeError("手柄采集（XInput）仅支持 Windows 实机；当前平台无法使用")
        self._user_index = user_index
        self._xinput = None
        errors: list[str] = []
        for dll_name in ("xinput1_4.dll", "xinput9_1_0.dll"):
            try:
                self._xinput = ctypes.windll.LoadLibrary(dll_name)
                break
            except OSError as exc:
                errors.append(f"{dll_name}: {exc}")
        if self._xinput is None:
            raise RuntimeError("XInput DLL 加载失败（" + "；".join(errors) + "）")
        self._get_state = self._xinput.XInputGetState

    def poll(self) -> tuple[int, int, int, int, int, int, int] | None:
        """读一次手柄状态 → (wButtons, LT, RT, LX, LY, RX, RY)。

        手柄未连接返回 None；其他 XInput 错误抛 RuntimeError 暴露问题。
        """
        state = XINPUT_STATE()
        result = self._get_state(self._user_index, ctypes.byref(state))
        if result == _ERROR_DEVICE_NOT_CONNECTED:
            return None
        if result != 0:
            raise RuntimeError(f"XInputGetState 失败: 错误码 {result}")
        g = state.Gamepad
        return (
            g.wButtons, g.bLeftTrigger, g.bRightTrigger,
            g.sThumbLX, g.sThumbLY, g.sThumbRX, g.sThumbRY,
        )


class GamepadCapture(QueuedInputCapture):
    """XInput 轮询线程 → ActionRecord 队列（InputCapture 协议）。

    poller 可注入（测试用假 poller，不触达真实 DLL）；默认构造 XInputPoller，
    非 Windows 在构造时即抛明确 RuntimeError。
    """

    def __init__(
        self,
        poller: XInputPoller | None = None,
        poll_hz: float = 120.0,
        deadzone: float = 0.15,
        epsilon: float = 0.02,
        trigger_threshold: float = 0.5,
        button_map: dict[str, str] | None = None,
        source: str = SOURCE_HUMAN,
        logger: logging.Logger | None = None,
    ):
        super().__init__()
        if poll_hz <= 0:
            raise ValueError(f"poll_hz 必须为正数: {poll_hz!r}")
        if not 0 <= deadzone < 1:
            raise ValueError(f"deadzone 必须在 [0, 1): {deadzone!r}")
        self._poller = poller if poller is not None else XInputPoller()
        self._poll_interval = 1.0 / poll_hz
        self._deadzone = deadzone
        self._epsilon = epsilon
        self._trigger_threshold = trigger_threshold
        self._button_map = button_map
        self._source = source
        self._log = logger or logging.getLogger("ai_game_player.input")
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_action: NormalizedAction | None = None
        self._started = False
        self._disconnect_logged = False

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                raw = self._poller.poll()
            except RuntimeError as exc:
                # 轮询失败：记日志后降级继续，不让采集线程静默死亡
                self._log.error("手柄轮询失败（继续重试）: %s", exc)
                self._stop_event.wait(self._poll_interval)
                continue
            if raw is None:
                if not self._disconnect_logged:
                    self._log.warning("手柄未连接，等待接入...")
                    self._disconnect_logged = True
                self._stop_event.wait(self._poll_interval)
                continue
            if self._disconnect_logged:
                self._log.info("手柄已接入")
                self._disconnect_logged = False

            timestamp_us = now_us()  # §11：收到该帧状态的时刻打点
            action = parse_gamepad_state(
                *raw,
                deadzone=self._deadzone,
                trigger_threshold=self._trigger_threshold,
                button_map=self._button_map,
            )
            if _action_changed(self._last_action, action, self._epsilon):
                self._last_action = action
                self._emit(ActionRecord(timestamp_us, action, self._source))
            self._stop_event.wait(self._poll_interval)

    def start(self) -> None:
        if self._started:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="gamepad-poll")
        self._thread.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
