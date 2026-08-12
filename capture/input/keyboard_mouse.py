"""键鼠输入采集：pynput 全局监听 → NormalizedAction 事件（spec §9/§10/§11）。

结构分两层：

- `KeyMouseMapper`：纯逻辑核心，平台无关、无 pynput 依赖，接收规范化后的
  键名字符串事件，维护状态（按下的移动方向键、按钮集、累计鼠标 delta），
  状态变化时产出 NormalizedAction 快照。单元测试全部打到这一层。
- `KeyboardMouseCapture`：InputCapture 实现（base.py 的 queue + start/stop
  线程模型），在 start() 时才延迟导入 pynput，把 pynput 回调事件翻译成
  规范化键名喂给 mapper，事件发生时用统一时钟打点（spec §11）后入队。

行为要点（spec §10）：
- move_forward/back/left/right 键合成二维移动向量；多键同按向量合成，
  长度超过 1 时归一化到单位长度（保持在 [-1,1]）。
- 其余映射键 / 鼠标键 → NormalizedAction buttons；未映射的键忽略。
- 鼠标移动累计 dx/dy，合并窗口（coalesce_ms）到期后归一化
  （norm_pixels 像素 = 满幅 1.0，clamp 到 [-1,1]）作为 camera_x/camera_y
  发出一次快照。
- 每次状态变化（按下/松开/鼠标合并窗口到期）发一个 source=human 快照。

键名规范化：单字符小写（"w"）；pynput 特殊键用 Key.name（"space"/"f12"）；
鼠标键统一 "mouse_left"/"mouse_right"/"mouse_middle"（与 config keys 段
的 mouse_* 前缀约定一致）。
"""
from __future__ import annotations

import logging
import math
import threading

from capture.action import SOURCE_HUMAN, ActionRecord, NormalizedAction
from capture.clock import now_us
from capture.input.base import QueuedInputCapture

# 移动方向动作 → (move_x 增量, move_y 增量)；move_y 前为正，move_x 右为正（§9）
MOVE_DIRECTIONS: dict[str, tuple[float, float]] = {
    "move_forward": (0.0, 1.0),
    "move_back": (0.0, -1.0),
    "move_left": (-1.0, 0.0),
    "move_right": (1.0, 0.0),
}

_MOUSE_BUTTON_NAMES = ("left", "right", "middle")


def canonical_key(name: str) -> str:
    """规范化键名：去首尾空白并小写（"W"/"w"、 "F12"/"f12" 视为同键）。"""
    return name.strip().lower()


def build_reverse_keymap(keys: dict[str, str]) -> dict[str, str]:
    """由 config keys 段（动作 → 键位）推出反向映射（规范化键名 → 动作名）。

    同一键位映射到多个动作属于配置错误，直接抛 ValueError 暴露。
    """
    reverse: dict[str, str] = {}
    for action, key in keys.items():
        canon = canonical_key(key)
        existing = reverse.get(canon)
        if existing is not None and existing != action:
            raise ValueError(f"键位冲突: {key!r} 同时映射到 {existing!r} 和 {action!r}")
        reverse[canon] = action
    return reverse


class KeyMouseMapper:
    """键鼠事件 → NormalizedAction 快照的纯逻辑核心（无平台依赖）。"""

    def __init__(self, key_to_action: dict[str, str], norm_pixels: float = 50.0):
        if norm_pixels <= 0:
            raise ValueError(f"norm_pixels 必须为正数: {norm_pixels!r}")
        self._key_to_action = {canonical_key(k): v for k, v in key_to_action.items()}
        self._norm_pixels = float(norm_pixels)
        self._move_dirs: set[str] = set()  # 当前按下的移动方向动作名
        self._buttons: set[str] = set()  # 当前按下的动作按钮
        self._mouse_dx = 0.0  # 合并窗口内累计的鼠标 delta（像素）
        self._mouse_dy = 0.0

    @property
    def has_pending_camera(self) -> bool:
        return self._mouse_dx != 0.0 or self._mouse_dy != 0.0

    def on_key(self, key_name: str, pressed: bool) -> NormalizedAction | None:
        """键盘按下/松开事件。未映射的键忽略（返回 None）。"""
        action = self._key_to_action.get(canonical_key(key_name))
        if action is None:
            return None
        if action in MOVE_DIRECTIONS:
            (self._move_dirs.add if pressed else self._move_dirs.discard)(action)
        else:
            (self._buttons.add if pressed else self._buttons.discard)(action)
        return self._snapshot()

    def on_mouse_button(self, button_name: str, pressed: bool) -> NormalizedAction | None:
        """鼠标键按下/松开事件。button_name 形如 "mouse_left"。"""
        return self.on_key(button_name, pressed)

    def on_mouse_move(self, dx: float, dy: float) -> None:
        """鼠标移动事件：累计 delta，等合并窗口到期统一发出。"""
        self._mouse_dx += dx
        self._mouse_dy += dy

    def flush_camera(self) -> NormalizedAction | None:
        """合并窗口到期：把累计鼠标 delta 归一化为 camera 轴并发快照。

        无累计 delta 时返回 None。camera_x 右转为正；屏幕坐标 dy 向下为正，
        取反使 camera_y 上抬为正（§9 定义）。
        """
        if not self.has_pending_camera:
            return None
        camera_x = self._mouse_dx / self._norm_pixels
        camera_y = -self._mouse_dy / self._norm_pixels
        self._mouse_dx = 0.0
        self._mouse_dy = 0.0
        return self._snapshot(camera_x=camera_x, camera_y=camera_y)

    def _move_vector(self) -> tuple[float, float]:
        x = sum(MOVE_DIRECTIONS[d][0] for d in self._move_dirs)
        y = sum(MOVE_DIRECTIONS[d][1] for d in self._move_dirs)
        length = math.hypot(x, y)
        if length > 1.0:  # 多键同按（如 W+D）向量归一化到单位长度
            x /= length
            y /= length
        return x, y

    def _snapshot(self, camera_x: float = 0.0, camera_y: float = 0.0) -> NormalizedAction:
        move_x, move_y = self._move_vector()
        return NormalizedAction(
            move_x=move_x,
            move_y=move_y,
            camera_x=camera_x,
            camera_y=camera_y,
            buttons=frozenset(self._buttons),
        )


def _pynput_key_to_str(key) -> str | None:
    """pynput 键对象 → 规范化键名（鸭子类型，不 import pynput，便于测试）。

    KeyCode 取 .char；特殊键取 .name（pynput Key 枚举成员带 name 属性）。
    """
    char = getattr(key, "char", None)
    if isinstance(char, str) and char:
        return canonical_key(char)
    name = getattr(key, "name", None)
    if isinstance(name, str) and name:
        return canonical_key(name)
    return None


def _pynput_mouse_button_to_str(button) -> str | None:
    """pynput 鼠标 Button → "mouse_left"/"mouse_right"/"mouse_middle"。"""
    name = getattr(button, "name", None)
    if isinstance(name, str) and name in _MOUSE_BUTTON_NAMES:
        return f"mouse_{name}"
    return None


class KeyboardMouseCapture(QueuedInputCapture):
    """pynput 全局键鼠监听器，事件即 ActionRecord(source=human) 入队。

    pynput 延迟到 start() 才导入：Linux 开发机无 pynput 也能 import 本模块、
    跑 mapper 层全部单元测试；缺依赖时 start() 抛带安装指引的 RuntimeError。
    """

    def __init__(
        self,
        key_to_action: dict[str, str],
        norm_pixels: float = 50.0,
        coalesce_ms: float = 10.0,
        source: str = SOURCE_HUMAN,
        logger: logging.Logger | None = None,
    ):
        super().__init__()
        if coalesce_ms <= 0:
            raise ValueError(f"coalesce_ms 必须为正数: {coalesce_ms!r}")
        self._mapper = KeyMouseMapper(key_to_action, norm_pixels)
        self._coalesce_s = coalesce_ms / 1000.0
        self._source = source
        self._log = logger or logging.getLogger("ai_game_player.input")
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._kb_listener = None
        self._mouse_listener = None
        self._coalesce_thread: threading.Thread | None = None
        self._last_pos: tuple[float, float] | None = None
        self._started = False

    # ---- pynput 回调（在 pynput 监听线程执行） ----

    def _on_key_event(self, key, pressed: bool) -> None:
        timestamp_us = now_us()  # §11：事件回调入口立刻打点
        key_name = _pynput_key_to_str(key)
        if key_name is None:
            return
        with self._lock:
            action = self._mapper.on_key(key_name, pressed)
        if action is not None:
            self._emit(ActionRecord(timestamp_us, action, self._source))

    def _on_click(self, x, y, button, pressed: bool) -> None:  # noqa: ARG002 pynput 回调签名
        timestamp_us = now_us()
        button_name = _pynput_mouse_button_to_str(button)
        if button_name is None:
            return
        with self._lock:
            action = self._mapper.on_mouse_button(button_name, pressed)
        if action is not None:
            self._emit(ActionRecord(timestamp_us, action, self._source))

    def _on_move(self, x: float, y: float) -> None:
        with self._lock:
            if self._last_pos is None:
                self._last_pos = (x, y)
                return  # 首个移动事件只建基准点，无 delta 可算
            dx = x - self._last_pos[0]
            dy = y - self._last_pos[1]
            self._last_pos = (x, y)
            self._mapper.on_mouse_move(dx, dy)

    def _coalesce_loop(self) -> None:
        """鼠标合并窗口：周期检查累计 delta，到期即归一化发出 camera 快照。"""
        while not self._stop_event.wait(self._coalesce_s):
            timestamp_us = now_us()
            with self._lock:
                action = self._mapper.flush_camera()
            if action is not None:
                self._emit(ActionRecord(timestamp_us, action, self._source))

    # ---- InputCapture 生命周期 ----

    def start(self) -> None:
        if self._started:
            return
        try:
            from pynput import keyboard, mouse  # 延迟导入：开发机可无 pynput
        except ImportError as exc:
            raise RuntimeError(
                "未安装 pynput（pyproject 已声明该依赖）；请执行: pip install pynput"
            ) from exc

        self._stop_event.clear()
        self._kb_listener = keyboard.Listener(
            on_press=lambda key: self._on_key_event(key, True),
            on_release=lambda key: self._on_key_event(key, False),
        )
        self._mouse_listener = mouse.Listener(
            on_move=self._on_move,
            on_click=self._on_click,
        )
        self._coalesce_thread = threading.Thread(
            target=self._coalesce_loop, daemon=True, name="km-coalesce"
        )
        self._kb_listener.start()
        self._mouse_listener.start()
        self._coalesce_thread.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        self._stop_event.set()
        for listener in (self._kb_listener, self._mouse_listener):
            if listener is not None:
                try:
                    listener.stop()
                except Exception as exc:  # 监听器停止失败不掩盖，只记日志降级
                    self._log.warning("pynput 监听器停止异常（忽略继续退出）: %s", exc)
        if self._coalesce_thread is not None:
            self._coalesce_thread.join(timeout=1.0)
        self._kb_listener = None
        self._mouse_listener = None
