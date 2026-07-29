"""控制层接口契约：输入模拟。

合规约束（计划文档第 2 节）：仅模拟键鼠输入，不读内存、不注入。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from core.contracts import Action


@dataclass
class Result:
    """动作执行结果。"""

    success: bool
    detail: str = ""


@runtime_checkable
class Controller(Protocol):
    """输入执行器。只认识 Action 契约，不认识具体游戏。"""

    def execute(self, action: Action) -> Result: ...
