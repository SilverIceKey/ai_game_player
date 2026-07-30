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
    move_hold_seconds: float = 0.35  # tap 模式下移动键按住时长
    action_pause: float = 0.05  # 每次输入后的间隔，避免输入洪峰
    turn_degrees_per_second: float = 180.0  # 转向角速度（度/秒），平滑转向用
    turn_step_interval: float = 0.02  # 平滑转向的步进间隔（秒）
    move_mode: str = "hold"  # hold=持续按住（顺畅，实机反馈后改默认）/ tap=每 tick 点按（旧行为）


class DirectInputController:
    """实现 core.control.base.Controller 契约：把 Action 翻译成键鼠输入。"""

    def __init__(self, keymap: dict[str, str], params: ControlParams | None = None):
        self.keymap = dict(keymap)
        self.params = params or ControlParams()
        self._pdi = None
        self._pdi_failed = False
        self._held_move: set[str] = set()  # hold 模式下当前按住的移动键

    def _backend(self):
        if self._pdi is None and not self._pdi_failed:
            try:
                import pydirectinput  # 延迟导入：仅 Windows 实机可用

                self._pdi = pydirectinput
            except Exception:
                self._pdi_failed = True
        return self._pdi

    def execute(self, action: Action) -> Result:
        pdi = self._backend()
        if action.name == "idle":
            # 停止指令：松开所有按住的移动键
            if pdi is not None:
                self._release_move(pdi)
            return Result(True, "idle")
        if pdi is None:
            return Result(False, "pydirectinput 不可用（仅 Windows 实机支持输入模拟）")
        try:
            return self._dispatch(pdi, action)
        except KeyError as exc:
            return Result(False, f"键位未配置: {exc}")
        except Exception as exc:  # 输入层失败不应炸掉主循环
            return Result(False, f"输入执行失败: {exc}")

    def release_all(self) -> None:
        """松开全部按住的键。会话结束（含 Ctrl+C）必须调用，否则按键会卡死在按下态。"""
        pdi = self._backend()
        if pdi is not None:
            self._release_move(pdi)

    def _release_move(self, pdi) -> None:
        for key in list(self._held_move):
            try:
                pdi.keyUp(key)
            except Exception:
                pass
            self._held_move.discard(key)

    def _hold_move(self, pdi, key: str) -> None:
        """hold 模式：目标键按住不动；方向切换时先松旧键。同方向重复调用是 no-op。"""
        for held in list(self._held_move):
            if held != key:
                pdi.keyUp(held)
                self._held_move.discard(held)
        if key not in self._held_move:
            pdi.keyDown(key)
            self._held_move.add(key)

    def _dispatch(self, pdi, action: Action) -> Result:
        p = self.params
        name = action.name
        if name == "move":
            direction = action.params.get("direction", "forward")
            key = self.keymap[f"move_{direction}"]
            if p.move_mode == "tap":
                pdi.keyDown(key)
                time.sleep(p.move_hold_seconds)
                pdi.keyUp(key)
            else:
                self._hold_move(pdi, key)
        elif name == "turn":
            degrees = float(action.params.get("degrees", 0.0))
            sign = -1.0 if action.params.get("direction") == "left" else 1.0
            self._smooth_turn(pdi, degrees * sign)
        elif name in ("light_attack", "dodge", "heal", "lock_on"):
            self._tap(pdi, self.keymap[name])
        else:
            return Result(False, f"未知动作: {name}")
        if p.action_pause > 0:
            time.sleep(p.action_pause)
        return Result(True, name)

    def _smooth_turn(self, pdi, signed_degrees: float) -> None:
        """线性平滑转向：按角速度把总像素拆成小步 moveRel，步间 sleep。

        一次性 moveRel 整段像素在游戏中表现为镜头猛甩（实机反馈"跳转"），
        拆步后接近真人甩鼠标的连续转动。像素取整的残差逐步累积，总量精确。
        """
        p = self.params
        total_px = signed_degrees * p.pixels_per_degree
        duration = abs(signed_degrees) / max(p.turn_degrees_per_second, 1e-6)
        steps = max(1, int(round(duration / p.turn_step_interval)))
        per_step = total_px / steps
        acc = 0.0
        for i in range(steps):
            acc += per_step
            whole = int(round(acc))
            if whole != 0:
                pdi.moveRel(whole, 0, relative=True)
                acc -= whole
            if i < steps - 1:
                time.sleep(p.turn_step_interval)

    @staticmethod
    def _tap(pdi, key: str) -> None:
        if key.startswith("mouse_"):
            pdi.click(button=key.removeprefix("mouse_"))
        else:
            pdi.press(key)
