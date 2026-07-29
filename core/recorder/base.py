"""记录层接口契约：战斗日志与回放样本。

回放产物是 LLM 离线复盘（llm/review）的唯一输入。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from core.contracts import Action, GameState, Suggestion


@dataclass
class StepRecord:
    """单步记录：状态 → 决策输出 → 执行结果。"""

    timestamp: float
    state: GameState
    output: Action | Suggestion
    result: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Recorder(Protocol):
    """回放记录器。全自动/半自动模式共用。"""

    def record(self, step: StepRecord) -> None: ...

    def export(self) -> Path:
        """导出本次会话的回放文件路径。"""
        ...
