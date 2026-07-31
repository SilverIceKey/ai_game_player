"""安全层（M3 计划 2.4；规格书 §14/18）：F12 全局急停 + 窗口失焦保护。

- 急停：ctypes GetAsyncKeyState 轮询（无新依赖），按下即停并释放全部按键；
  恢复交互 = 急停中再按一次急停键（toggle，边沿检测），每次翻转进日志。
- 失焦：GetForegroundWindow 对比游戏窗口 hwnd（复用 foreground 的窗口枚举），
  失焦立即释放输入并暂停决策（本 tick 动作被仲裁阻断）。
- 平台约束：win32 调用全部隔离在默认 poller/checker 闭包里（非 Windows 返回安全默认），
  测试通过注入假 poller/checker 全覆盖，不触达真实 win32。
"""
from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass

from core.config import ConfigError


@dataclass(frozen=True)
class SafetyParams:
    emergency_stop_key: str = "F12"  # 全局急停键（F1-F12 或单字符键名）
    stop_on_focus_lost: bool = True  # 游戏窗口失焦立即释放输入并暂停决策


@dataclass(frozen=True)
class SafetyState:
    emergency_stopped: bool
    focus_lost: bool
    detail: str = ""  # 当前安全状态说明（空 = 正常）


def vk_for_key(key: str) -> int:
    """键名 → Win32 虚拟键码。支持 F1-F12 与单字符键名，其余报 ConfigError。"""
    name = key.strip().upper()
    if name.startswith("F") and name[1:].isdigit() and 1 <= int(name[1:]) <= 12:
        return 0x70 + int(name[1:]) - 1
    if len(name) == 1 and name.isalnum():
        return ord(name)
    raise ConfigError(f"safety.emergency_stop_key 无法识别: {key!r}（支持 F1-F12 或单字符键名）")


def _default_key_poller() -> Callable[[str], bool]:
    """默认急停键轮询（win32 GetAsyncKeyState）；非 Windows 恒为未按下。"""
    if sys.platform != "win32":
        return lambda key: False

    def _poll(key: str) -> bool:
        import ctypes

        return bool(ctypes.windll.user32.GetAsyncKeyState(vk_for_key(key)) & 0x8000)

    return _poll


def _default_focus_checker(window_title: str) -> Callable[[], bool]:
    """默认焦点检查（游戏窗口是否前台）；非 Windows 恒为前台。"""
    if sys.platform != "win32":
        return lambda: True

    hwnd_cache: dict[str, int] = {}

    def _focused() -> bool:
        import ctypes

        from core.perception.foreground import list_window_handles
        from core.perception.mss_source import match_window_title

        hwnd = hwnd_cache.get("hwnd")
        if hwnd is None:
            matched = match_window_title(window_title, [t for t, _ in list_window_handles()])
            if matched is None:
                return False  # 窗口都找不到，视同失焦
            hwnd = hwnd_cache["hwnd"] = dict(list_window_handles())[matched]
        return ctypes.windll.user32.GetForegroundWindow() == hwnd

    return _focused


class SafetyMonitor:
    """每 tick 轮询急停键与窗口焦点。急停/失焦时回调释放输入。"""

    def __init__(
        self,
        params: SafetyParams | None = None,
        window_title: str = "",
        on_release: Callable[[], None] | None = None,
        key_poller: Callable[[str], bool] | None = None,
        focus_checker: Callable[[], bool] | None = None,
    ):
        self.params = params or SafetyParams()
        vk_for_key(self.params.emergency_stop_key)  # 构造期校验键名，非法即报错
        self.window_title = window_title
        self._on_release = on_release
        self._key_poller = key_poller or _default_key_poller()
        self._focus_checker = focus_checker or _default_focus_checker(window_title)
        self._stopped = False
        self._key_was_down = False

    @property
    def emergency_stopped(self) -> bool:
        return self._stopped

    def check(self) -> SafetyState:
        # 急停键边沿检测（toggle）：按下进入急停，急停中再按一次恢复
        down = bool(self._key_poller(self.params.emergency_stop_key))
        if down and not self._key_was_down:
            self._stopped = not self._stopped
        self._key_was_down = down

        if self._stopped:
            self._release()
            return SafetyState(True, False, f"人工急停中（按 {self.params.emergency_stop_key} 恢复）")

        if self.params.stop_on_focus_lost and not self._focus_checker():
            self._release()
            return SafetyState(False, True, "游戏窗口失焦，已释放输入并暂停决策")

        return SafetyState(False, False, "")

    def _release(self) -> None:
        if self._on_release is not None:
            try:
                self._on_release()
            except Exception:
                pass  # 释放失败不阻断安全状态本身
