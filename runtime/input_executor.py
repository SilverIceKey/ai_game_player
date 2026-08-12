"""KeyboardMouseExecutor（spec §9/§10：NormalizedAction → 实际键鼠输入）。

由旧 core/control/directinput.py 改造：动作源从离散 Action 换成
NormalizedAction（4 连续轴 + 按钮集），键位映射全部来自 config keys（§9
内部动作表示与实际键位解耦）。

- move 轴（超 move_deadzone）→ move_forward/back/left/right 键持续按住，
  按"按住集合差分"执行：新出现的方向 keyDown、消失的方向 keyUp，
  同方向重复 execute 是 no-op（避免反复敲击）。
- buttons → keymap 查找后同样做按下/松开差分；键位以 mouse_ 前缀表示鼠标键，
  用 pdi.mouseDown/mouseUp，其余用 pdi.keyDown/keyUp。
- camera 轴 → moveRel(round(camera_x*pixels_per_unit), round(-camera_y*pixels_per_unit))；
  camera_y 上为正（spec §9 视角上抬为正）对应鼠标向上移动（屏幕坐标 y 向下为正，取负）。
- keymap 缺失的动作名：不炸主循环，在执行明细中以 error 段返回。

平台约束：pydirectinput 延迟导入（首次 execute 才 import），Linux 开发机
不安装该包也可跑全部单元测试；后端可整体注入（测试用假 backend）。
"""
from __future__ import annotations

import time
from collections.abc import Callable

from capture.action import BUTTONS, NormalizedAction
from config import ExecutorConfig

# move 轴方向 → keymap 动作名（move_y 前为正、move_x 右为正）
_MOVE_DIRECTIONS: tuple[tuple[str, str], ...] = (
    ("move_forward", "move_y+"),
    ("move_back", "move_y-"),
    ("move_right", "move_x+"),
    ("move_left", "move_x-"),
)


class KeyboardMouseExecutor:
    """NormalizedAction → 键鼠输入执行器。线程安全由调用方（input 线程独占）保证。"""

    def __init__(
        self,
        keymap: dict[str, str],
        params: ExecutorConfig | None = None,
        backend: object | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._keymap = dict(keymap)
        self._params = params or ExecutorConfig()
        self._pdi = backend
        self._pdi_failed = False
        self._sleep = sleep
        self._held_moves: set[str] = set()  # 当前按住的移动键（keymap 动作名）
        self._held_buttons: set[str] = set()  # 当前按住的按钮（动作名）

    def _backend(self):
        """首次使用时延迟导入 pydirectinput；导入失败缓存失败态，不再重试。"""
        if self._pdi is None and not self._pdi_failed:
            try:
                import pydirectinput  # 延迟导入：仅 Windows 实机可用

                self._pdi = pydirectinput
            except ImportError:
                self._pdi_failed = True
        return self._pdi

    def execute(self, action: NormalizedAction) -> str:
        """执行一步动作，返回明细描述（日志/调试用）；输入层失败降级为 error 明细。"""
        pdi = self._backend()
        if pdi is None:
            return "error: pydirectinput 不可用（仅 Windows 实机支持输入模拟）"

        details: list[str] = []
        errors: list[str] = []
        self._apply_move(pdi, action, details, errors)
        self._apply_buttons(pdi, action, details, errors)
        self._apply_camera(pdi, action, details)

        if self._params.action_pause > 0:
            self._sleep(self._params.action_pause)
        if errors:
            details.append("error: " + "; ".join(errors))
        return "; ".join(details) if details else "idle"

    def release_all(self) -> None:
        """松开全部按住的键与鼠标键（会话结束/接管/急停必须调用，§40）。

        逐个尽力释放：单个释放失败不阻断其余输入的释放。
        """
        pdi = self._backend()
        if pdi is None:
            return
        for name in list(self._held_moves) + list(self._held_buttons):
            key = self._keymap.get(name)
            if key is None:
                self._held_moves.discard(name)
                self._held_buttons.discard(name)
                continue
            try:
                self._release_key(pdi, key)
            except Exception:
                pass  # 释放是安全兜底：尽力而为，不在此处抛错
            self._held_moves.discard(name)
            self._held_buttons.discard(name)

    # ---------- move 轴 → 方向键差分 ----------

    def _desired_moves(self, action: NormalizedAction) -> set[str]:
        deadzone = self._params.move_deadzone
        desired: set[str] = set()
        for name, axis in _MOVE_DIRECTIONS:
            value = action.move_y if axis.startswith("move_y") else action.move_x
            if axis.endswith("+") and value > deadzone:
                desired.add(name)
            elif axis.endswith("-") and value < -deadzone:
                desired.add(name)
        return desired

    def _apply_move(
        self, pdi, action: NormalizedAction, details: list[str], errors: list[str]
    ) -> None:
        desired = self._desired_moves(action)
        for name in sorted(desired | self._held_moves):
            key = self._keymap.get(name)
            if key is None:
                if name in desired:
                    errors.append(f"键位未配置: {name}")
                continue
            if name in desired and name not in self._held_moves:
                pdi.keyDown(key)
                self._held_moves.add(name)
                details.append(f"hold:{key}")
            elif name not in desired and name in self._held_moves:
                pdi.keyUp(key)
                self._held_moves.discard(name)
                details.append(f"release:{key}")

    # ---------- buttons → 键/鼠标键差分 ----------

    def _apply_buttons(
        self, pdi, action: NormalizedAction, details: list[str], errors: list[str]
    ) -> None:
        # 遍历顺序固定为 spec §9 按钮表，保证明细与按键顺序确定
        for name in BUTTONS:
            pressed = name in action.buttons
            held = name in self._held_buttons
            if pressed == held:
                continue
            key = self._keymap.get(name)
            if key is None:
                if pressed:
                    errors.append(f"键位未配置: {name}")
                continue
            if pressed:
                self._press_key(pdi, key)
                self._held_buttons.add(name)
                details.append(f"press:{key}")
            else:
                self._release_key(pdi, key)
                self._held_buttons.discard(name)
                details.append(f"release:{key}")

    # ---------- camera 轴 → 鼠标相对移动 ----------

    def _apply_camera(self, pdi, action: NormalizedAction, details: list[str]) -> None:
        dx = round(action.camera_x * self._params.pixels_per_unit)
        dy = round(-action.camera_y * self._params.pixels_per_unit)  # 上抬为正 → 鼠标上移
        if dx != 0 or dy != 0:
            pdi.moveRel(dx, dy, relative=True)
            details.append(f"camera:dx={dx},dy={dy}")

    # ---------- 按键原语 ----------

    @staticmethod
    def _press_key(pdi, key: str) -> None:
        if key.startswith("mouse_"):
            pdi.mouseDown(button=key.removeprefix("mouse_"))
        else:
            pdi.keyDown(key)

    @staticmethod
    def _release_key(pdi, key: str) -> None:
        if key.startswith("mouse_"):
            pdi.mouseUp(button=key.removeprefix("mouse_"))
        else:
            pdi.keyUp(key)
