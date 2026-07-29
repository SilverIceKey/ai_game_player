"""通用有限状态机引擎（游戏无关）。

状态与转移由各游戏在 games/<game>/ 中定义（如 games/wukong/combat.py），
本模块只负责注册与求值：每 tick 最多一次转移，全局转移优先于普通转移。
"""
from __future__ import annotations

from collections.abc import Callable

from core.contracts import GameState

Condition = Callable[[GameState], bool]


class StateMachine:
    def __init__(self, initial: str):
        if not initial:
            raise ValueError("初始状态不能为空")
        self._current = initial
        self._transitions: list[tuple[str, str, Condition]] = []
        self._global: list[tuple[str, Condition]] = []

    @property
    def current(self) -> str:
        return self._current

    def add_transition(self, src: str, dst: str, condition: Condition) -> None:
        """普通转移：仅当当前状态为 src 且条件成立时生效，按注册顺序求值。"""
        self._transitions.append((src, dst, condition))

    def add_global_transition(self, dst: str, condition: Condition) -> None:
        """全局转移：任意状态（除 dst 自身）条件成立即生效，优先于普通转移。"""
        self._global.append((dst, condition))

    def update(self, state: GameState) -> bool:
        """求值一次，返回本 tick 是否发生了状态转移。"""
        for dst, cond in self._global:
            if dst != self._current and cond(state):
                self._current = dst
                return True
        for src, dst, cond in self._transitions:
            if src == self._current and dst != self._current and cond(state):
                self._current = dst
                return True
        return False
