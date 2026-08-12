"""闭环评估指标骨架（spec §37 Closed-loop Evaluation + §43 MVP 成功标准）。

§37 第一阶段主指标：
- Autonomous Duration（平均自主时长）
- Manual Takeover Rate（人工接管率）
- Stuck Rate（卡住时间占比）

§43 MVP 参考线：Autonomous Runtime >= 10 min、Takeover Rate 持续下降、
Stuck Ratio < 10%。

时间戳一律使用 capture.clock.now_us() 的统一时钟（spec §11），
由调用方传入，本类不自行取时，保持纯逻辑可测。
"""
from __future__ import annotations

from typing import Any

_US_PER_HOUR = 3_600_000_000.0


class ClosedLoopMetrics:
    """AUTOPILOT 闭环运行指标累计器。

    用法：
        m.start_autonomous(now_us())   # AI 开始自主控制
        m.record_takeover(now_us())    # 人工接管（spec §26）
        m.start_autonomous(now_us())   # 接管结束，恢复自主
        m.record_stuck(now_us(), 3000) # 检测到卡住 3000ms
        m.stop(now_us())               # 本次闭环结束
    """

    def __init__(self) -> None:
        self._window_start_us: int | None = None  # 闭环窗口起点（首次 start_autonomous）
        self._window_end_us: int | None = None  # stop 时刻
        self._autonomous_start_us: int | None = None  # 当前自主段起点
        self._autonomous_ms = 0.0  # 累计自主时长
        self._takeover_count = 0
        self._stuck_ms = 0.0

    @property
    def is_autonomous(self) -> bool:
        return self._autonomous_start_us is not None

    def start_autonomous(self, now_us: int) -> None:
        """AI 开始（或恢复）自主控制。"""
        if self._window_start_us is None:
            self._window_start_us = now_us
        if self._autonomous_start_us is not None:
            raise RuntimeError("已处于自主控制状态，重复 start_autonomous")
        self._autonomous_start_us = now_us

    def record_takeover(self, now_us: int) -> None:
        """人工接管：累计当前自主段时长，结束自主状态。"""
        if self._autonomous_start_us is None:
            raise RuntimeError("当前不在自主控制状态，不能记录 takeover")
        self._autonomous_ms += (now_us - self._autonomous_start_us) / 1000.0
        self._autonomous_start_us = None
        self._takeover_count += 1

    def record_stuck(self, now_us: int, duration_ms: float) -> None:
        """记录一次卡住事件（时长毫秒）。now_us 保留给事件日志扩展。"""
        if duration_ms < 0:
            raise ValueError(f"stuck duration_ms 不能为负: {duration_ms}")
        self._stuck_ms += float(duration_ms)

    def stop(self, now_us: int) -> None:
        """结束本次闭环：收尾未结的自主段，冻结窗口。"""
        if self._autonomous_start_us is not None:
            self._autonomous_ms += (now_us - self._autonomous_start_us) / 1000.0
            self._autonomous_start_us = None
        self._window_end_us = now_us

    def summary(self) -> dict[str, Any]:
        """汇总主指标（spec §37/§43）。

        - autonomous_duration_ms：累计自主控制时长
        - takeover_count / takeover_rate_per_hour：接管次数与每小时接管率
          （按自主时长归一；自主时长为 0 时记 0）
        - stuck_ratio：卡住时长 / 闭环窗口总时长（窗口未结束时按当前累计事件
          无法确定窗口，记 0）
        """
        autonomous_hours = self._autonomous_ms / _US_PER_HOUR * 1000.0
        takeover_rate = self._takeover_count / autonomous_hours if autonomous_hours > 0 else 0.0
        window_ms = (
            (self._window_end_us - self._window_start_us) / 1000.0
            if self._window_start_us is not None and self._window_end_us is not None
            else 0.0
        )
        stuck_ratio = self._stuck_ms / window_ms if window_ms > 0 else 0.0
        return {
            "autonomous_duration_ms": self._autonomous_ms,
            "takeover_count": self._takeover_count,
            "takeover_rate_per_hour": takeover_rate,
            "stuck_ms": self._stuck_ms,
            "stuck_ratio": stuck_ratio,
            "window_duration_ms": window_ms,
        }
