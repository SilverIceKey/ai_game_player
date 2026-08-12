"""Safety Filter（spec §39 Safety Filter + §40 Dead Man Switch + §26 Human Override + §47 失败回退）。

由旧 core/safety.py（F12 急停 toggle + 失焦保护）扩展迁移：

- 状态机（§26）：AI_CONTROL ⇄ HUMAN_OVERRIDE，override 键 toggle 边沿检测切换。
  进入 HUMAN_OVERRIDE 立即触发 Dead Man Switch（清空 Action Queue + 释放全部输入），
  接管期间 filter_action 阻断全部动作。
- 失焦保护（§39）：游戏窗口必须前台，失焦立即 STOP ACTION 并释放输入。
- 输出约束（§39，作用于 NormalizedAction）：
  camera 轴单步超 max_camera_delta 截断；单按钮连续按住超 max_button_hold_ms
  自动释放（从动作中移除，直到模型先松手才允许再次按下）；动作频率超
  max_action_rate_hz 丢弃；NaN/非有限值等异常模型输出直接丢弃。
  （键级黑名单如 Win Key / Alt+F4 不在本层：本层只见归一化动作，键位合法性
  由 config keys 校验与 executor 映射负责。）
- Dead Man Switch（§40/§47）：语义 = executor 释放全部输入 + scheduler 清空队列，
  本模块不直接持有两者，通过注入回调执行。

平台约束：win32（GetAsyncKeyState / GetForegroundWindow / 窗口枚举）全部延迟导入
且可注入；非 Windows 平台默认轮询恒为"未按下"、焦点恒为"前台"，测试注入假
poller/checker 全覆盖，不触达真实 win32。
"""
from __future__ import annotations

import math
import sys
from collections.abc import Callable
from dataclasses import dataclass

from capture.action import NormalizedAction
from config import SafetyConfig

# spec §26 顶层控制状态
MODE_AI_CONTROL = "AI_CONTROL"
MODE_HUMAN_OVERRIDE = "HUMAN_OVERRIDE"


@dataclass(frozen=True)
class SafetyState:
    """一次环境检查的结果（detail 为空 = 正常，AI 可执行动作）。"""

    mode: str
    override_active: bool
    focus_lost: bool
    detail: str = ""


def vk_for_key(key: str) -> int:
    """键名 → Win32 虚拟键码。支持 F1-F12 与单字符键名，其余报 ValueError。"""
    name = key.strip().upper()
    if name.startswith("F") and name[1:].isdigit() and 1 <= int(name[1:]) <= 12:
        return 0x70 + int(name[1:]) - 1
    if len(name) == 1 and name.isalnum():
        return ord(name)
    raise ValueError(f"safety.override_key 无法识别: {key!r}（支持 F1-F12 或单字符键名）")


def _match_window_title(query: str, titles: list[str]) -> str | None:
    """窗口标题匹配：先精确（忽略大小写与首尾空格），再包含匹配。"""
    exact = query.strip().casefold()
    for title in titles:
        if title.strip().casefold() == exact:
            return title
    for title in titles:
        if exact in title.strip().casefold():
            return title
    return None


def _default_key_poller() -> Callable[[str], bool]:
    """默认 override 键轮询（win32 GetAsyncKeyState）；非 Windows 恒为未按下。"""
    if sys.platform != "win32":
        return lambda key: False

    def _poll(key: str) -> bool:
        import ctypes

        return bool(ctypes.windll.user32.GetAsyncKeyState(vk_for_key(key)) & 0x8000)

    return _poll


def _default_focus_checker(window_title: str) -> Callable[[], bool]:
    """默认焦点检查（游戏窗口是否前台）；非 Windows 恒为前台。

    窗口枚举延迟到调用处 import capture.screen.foreground：该模块由采集侧
    维护且仅 Windows 可用，此处不触发静态导入失败。
    """
    if sys.platform != "win32":
        return lambda: True

    hwnd_cache: dict[str, int] = {}

    def _focused() -> bool:
        import ctypes

        from capture.screen.foreground import list_window_handles

        hwnd = hwnd_cache.get("hwnd")
        if hwnd is None:
            handles = list_window_handles()
            matched = _match_window_title(window_title, [t for t, _ in handles])
            if matched is None:
                return False  # 窗口都找不到，视同失焦
            hwnd = hwnd_cache["hwnd"] = dict(handles)[matched]
        return ctypes.windll.user32.GetForegroundWindow() == hwnd

    return _focused


class SafetyFilter:
    """AUTOPILOT 输出端的最后一道闸：环境检查 + 动作过滤。

    调用约定：主循环每个动作都走 filter_action（内部自带环境检查）；
    check_environment 同时暴露给 app 层，用于无动作流动时也要轮询
    override/失焦（否则接管状态下按键释放依赖下一次动作到达才发现）。
    """

    def __init__(
        self,
        safety: SafetyConfig,
        window_title: str = "",
        on_release: Callable[[], None] | None = None,
        on_clear: Callable[[], None] | None = None,
        key_poller: Callable[[str], bool] | None = None,
        focus_checker: Callable[[], bool] | None = None,
    ):
        self._safety = safety
        vk_for_key(safety.override_key)  # 构造期校验键名，非法即报错
        self._on_release = on_release  # 通常为 executor.release_all（§40 Release All Keys）
        self._on_clear = on_clear  # 通常为 scheduler.clear（§40 Clear Action Queue）
        self._key_poller = key_poller or _default_key_poller()
        self._focus_checker = focus_checker or _default_focus_checker(window_title)

        self._mode = MODE_AI_CONTROL
        self._key_was_down = False
        self._hold_start_us: dict[str, int] = {}  # 按钮连续按住的起始时刻
        self._force_released: set[str] = set()  # 已超长按被强制释放、等待模型先松手的按钮
        self._last_pass_us: int | None = None  # 上一个放行动作的时刻（限频用）
        self._min_interval_us = int(1000.0 / safety.max_action_rate_hz * 1000.0)
        self._max_hold_us = int(safety.max_button_hold_ms * 1000.0)

    @property
    def mode(self) -> str:
        return self._mode

    def dead_man_switch(self) -> None:
        """紧急停止（§40）：释放全部输入 + 清空动作队列，通过注入回调执行。

        回调异常不阻断其余回调（释放输入是安全兜底，必须尽力执行到底）。
        """
        errors: list[str] = []
        for name, callback in (("release", self._on_release), ("clear", self._on_clear)):
            if callback is None:
                continue
            try:
                callback()
            except Exception as exc:  # 安全兜底路径：记录但不中断另一个回调
                errors.append(f"{name}: {exc}")
        if errors:
            raise RuntimeError(f"dead man switch 回调执行失败: {'; '.join(errors)}")

    def check_environment(self) -> SafetyState:
        """轮询 override 键（toggle 边沿检测）与窗口焦点；不安全时触发 Dead Man Switch。"""
        down = bool(self._key_poller(self._safety.override_key))
        if down and not self._key_was_down:
            self._mode = (
                MODE_HUMAN_OVERRIDE if self._mode == MODE_AI_CONTROL else MODE_AI_CONTROL
            )
        self._key_was_down = down

        if self._mode == MODE_HUMAN_OVERRIDE:
            self.dead_man_switch()  # §26：接管必须立即清空队列 + 释放按键
            return SafetyState(
                self._mode, True, False,
                f"人工接管中（按 {self._safety.override_key} 恢复 AI 控制）",
            )

        focus_lost = self._safety.stop_on_focus_lost and not self._focus_checker()
        if focus_lost:
            self.dead_man_switch()  # §39：失焦立即 STOP ACTION
            return SafetyState(self._mode, False, True, "游戏窗口失焦，已停止动作并释放输入")

        return SafetyState(self._mode, False, False, "")

    def filter_action(self, action: NormalizedAction, now_us: int) -> NormalizedAction | None:
        """§39 输出约束。返回放行（可能修正过）的动作，或 None 表示阻断/丢弃。"""
        state = self.check_environment()
        if state.override_active or state.focus_lost:
            return None

        # 异常模型输出（NaN / inf）直接丢弃；NormalizedAction 的 clamp 会把 NaN
        # 静默夹成 ±1，必须先于一切修正判断非有限值
        axes = (action.move_x, action.move_y, action.camera_x, action.camera_y)
        if not all(math.isfinite(v) for v in axes):
            return None

        # 动作频率上限：到达间隔小于最小间隔的动作直接丢弃（不计入计时，
        # 让下一个到达较晚的动作有机会放行）
        if self._last_pass_us is not None and now_us - self._last_pass_us < self._min_interval_us:
            return None

        # camera 轴单步最大视角量截断
        limit = self._safety.max_camera_delta
        camera_x = max(-limit, min(limit, action.camera_x))
        camera_y = max(-limit, min(limit, action.camera_y))

        buttons = self._filter_button_holds(action.buttons, now_us)

        self._last_pass_us = now_us
        if camera_x == action.camera_x and camera_y == action.camera_y and buttons == action.buttons:
            return action
        return NormalizedAction(
            move_x=action.move_x,
            move_y=action.move_y,
            camera_x=camera_x,
            camera_y=camera_y,
            buttons=buttons,
        )

    def _filter_button_holds(self, buttons: frozenset[str], now_us: int) -> frozenset[str]:
        """单按钮连续按住超时自动释放（§39）。

        按住计时从按钮首次出现在放行流中开始；超时后该按钮持续从动作中剔除，
        直到模型输出中该按钮消失（视为松手）才允许下一次按下。
        """
        result: set[str] = set()
        for name in buttons:
            if name in self._force_released:
                continue  # 等待模型先松手
            start = self._hold_start_us.setdefault(name, now_us)
            if now_us - start > self._max_hold_us:
                self._force_released.add(name)  # 超时：本步起强制释放
                continue
            result.add(name)
        # 模型已松手的按钮：清零计时并解除强制释放
        for name in list(self._hold_start_us):
            if name not in buttons:
                del self._hold_start_us[name]
        self._force_released.intersection_update(buttons)
        return frozenset(result)
