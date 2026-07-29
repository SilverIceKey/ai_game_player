"""NullController：干跑（--dry-run）模式控制器。

满足 core.control.base.Controller Protocol，但绝不触达真实输入：
不 import pydirectinput，execute 直接返回失败占位结果。
首次实机启动时用它跑完整链路核对配置，安全无副作用。
"""
from __future__ import annotations

from core.contracts import Action
from core.control.base import Result


class NullController:
    """干跑控制器：记录链路照常走，输入一律不执行。"""

    def execute(self, action: Action) -> Result:
        return Result(success=False, detail="dry-run")
