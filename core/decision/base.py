"""决策层接口契约。

LLM 不在此链路中：实时决策只由规则/状态机/行为树承担，
LLM 仅离线复盘与调参（llm/ 模块）。
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from core.contracts import Action, GameState, Suggestion


@runtime_checkable
class DecisionEngine(Protocol):
    """决策引擎：输入标准状态，输出动作或建议。"""

    def decide(self, state: GameState) -> Action | Suggestion:
        """全自动模式返回 Action（交 control 执行）；半自动模式返回 Suggestion（交 UI 展示）。"""
        ...
