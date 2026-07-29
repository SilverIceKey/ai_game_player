"""游戏适配层接口契约。

每个游戏一个适配插件（games/<game>/），实现 GameAdapter。
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from core.contracts import Action, GameState


@runtime_checkable
class GameAdapter(Protocol):
    """游戏适配接口：把画面翻译成标准状态，并声明动作空间。"""

    def perceive(self, frame: np.ndarray) -> GameState:
        """把一帧画面解析为本游戏的标准状态。"""
        ...

    def available_actions(self, state: GameState) -> list[Action]:
        """当前状态下可执行的动作集。"""
        ...

    def action_space(self) -> list[str]:
        """本游戏的全部动作名（技能、移动、交互等）。"""
        ...
