"""统一 monotonic clock（spec §11：整个项目最高优先级之一）。

全项目唯一时间源。所有 Frame Capture / Input Capture / Inference /
Action Dispatch / Training Sample 的时间戳必须来自本模块，
才能精确回答"某一帧对应玩家哪一个操作"。

`time.perf_counter_ns` 在 Windows 上底层即 QueryPerformanceCounter（spec 推荐），
单调、不受系统时间调整影响。单位为微秒（us），与 spec §20 数据结构一致。
"""
from __future__ import annotations

import time


def now_us() -> int:
    """当前 monotonic 时间戳（微秒）。"""
    return time.perf_counter_ns() // 1000


def us_to_ms(timestamp_us: int) -> float:
    return timestamp_us / 1000.0


def ms_to_us(milliseconds: float) -> int:
    return int(milliseconds * 1000)
