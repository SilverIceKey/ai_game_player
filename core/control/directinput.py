"""DirectInputController：pydirectinput 键鼠输入执行器。

平台约束（计划文档第 4 节）：pydirectinput 延迟导入（首次 execute 时 import），
Linux 开发机不安装该包也可跑全部单元测试；Windows 实机才触达真实输入。
键位全部来自配置（configs/wukong.yaml），用户自定义键位实机校准时填入。
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from core.contracts import Action
from core.control.base import Result


@dataclass(frozen=True)
class ControlParams:
    pixels_per_degree: float = 12.0  # 鼠标像素 / 转角（实机校准）
    move_hold_seconds: float = 0.35  # 移动键按住时长
    action_pause: float = 0.05  # 每次输入后的间隔，避免输入洪峰


class DirectInputController:
    """实现 core.control.base.Controller 契约：把 Action 翻译成键鼠输入。"""

    def __init__(self, keymap: dict[str, str], params: ControlParams | None = None):
        self.keymap = dict(keymap)
        self.params = params or ControlParams()
        self._pdi = None
        self._pdi_failed = False

    def _backend(self):
        if self._pdi is None and not self._pdi_failed:
            try:
                import pydirectinput  # 延迟导入：仅 Windows 实机可用

                self._pdi = pydirectinput
            except Exception:
                self._pdi_failed = True
        return self._pdi

    def execute(self, action: Action) -> Result:
        if action.name == "idle":
            return Result(True, "idle")
        pdi = self._backend()
        if pdi is None:
            return Result(False, "pydirectinput 不可用（仅 Windows 实机支持输入模拟）")
        try:
            return self._dispatch(pdi, action)
        except KeyError as exc:
            return Result(False, f"键位未配置: {exc}")
        except Exception as exc:  # 输入层失败不应炸掉主循环
            return Result(False, f"输入执行失败: {exc}")

    def _dispatch(self, pdi, action: Action) -> Result:
        p = self.params
        name = action.name
        if name == "move":
            direction = action.params.get("direction", "forward")
            key = self.keymap[f"move_{direction}"]
            pdi.keyDown(key)
            time.sleep(p.move_hold_seconds)
            pdi.keyUp(key)
        elif name == "turn":
            degrees = float(action.params.get("degrees", 0.0))
            sign = -1.0 if action.params.get("direction") == "left" else 1.0
            pdi.moveRel(int(round(degrees * p.pixels_per_degree * sign)), 0, relative=True)
        elif name in ("light_attack", "dodge", "heal", "lock_on"):
            self._tap(pdi, self.keymap[name])
        else:
            return Result(False, f"未知动作: {name}")
        if p.action_pause > 0:
            time.sleep(p.action_pause)
        return Result(True, name)

    @staticmethod
    def _tap(pdi, key: str) -> None:
        if key.startswith("mouse_"):
            pdi.click(button=key.removeprefix("mouse_"))
        else:
            pdi.press(key)
