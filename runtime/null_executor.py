"""NullExecutor：dry-run / 测试用空执行器。

接口与 KeyboardMouseExecutor 对齐（execute/release_all），绝不触达真实输入：
不 import pydirectinput，只记录收到的动作，供链路联调与单元测试核对
"AI 现在想做什么"（同 spec §41 意图观察思路）。
"""
from __future__ import annotations

from capture.action import NormalizedAction


class NullExecutor:
    """空执行器：记录动作，不执行任何输入。"""

    def __init__(self) -> None:
        self.actions: list[NormalizedAction] = []
        self.release_all_calls = 0

    def execute(self, action: NormalizedAction) -> str:
        self.actions.append(action)
        if action.is_neutral():
            return "idle"
        return f"null: recorded ({len(self.actions)})"

    def release_all(self) -> None:
        self.release_all_calls += 1
