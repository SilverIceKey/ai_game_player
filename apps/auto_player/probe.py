"""--probe-input：输入链路诊断。

逐个动作发送真实输入并控制台播报，用户观察游戏画面确认哪些动作实际生效，
用于区分"决策没发动作"与"输入没送达游戏"两类根因
（如实机反馈：W 前进有效但 turn（合成鼠标相对移动）视角不转）。
"""
from __future__ import annotations

import time
from collections.abc import Callable

from core.contracts import Action
from core.control.base import Controller

# (播报描述, 动作)；顺序即执行顺序
PROBE_SEQUENCE: list[tuple[str, Action]] = [
    ("视角右转 30°（turn right）", Action("turn", {"degrees": 30.0, "direction": "right"})),
    ("视角左转 30°（turn left）", Action("turn", {"degrees": 30.0, "direction": "left"})),
    ("前进（move forward）", Action("move", {"direction": "forward"})),
    ("闪避（dodge）", Action("dodge")),
    ("轻棍攻击（light_attack）", Action("light_attack")),
    ("锁定/取消锁定（lock_on）", Action("lock_on")),
]


def run_probe(
    controller: Controller,
    countdown: float = 5.0,
    step_pause: float = 1.5,
    out: Callable[[str], None] = print,
) -> None:
    """按 PROBE_SEQUENCE 逐动作执行并播报。"""
    if countdown > 0:
        for remaining in range(int(countdown), 0, -1):
            out(f"[probe] {remaining}s 后开始（请确认游戏窗口在前台）……")
            time.sleep(1.0)
    for desc, action in PROBE_SEQUENCE:
        result = controller.execute(action)
        out(f"[probe] 已发送: {desc} → {result.detail or result.success}")
        if step_pause > 0:
            time.sleep(step_pause)
    out("[probe] 完成。请反馈：哪些动作在游戏中实际发生了，哪些没有？")
