"""evaluation/closed_loop.py 单元测试：闭环主指标手算（spec §37/§43）。"""
from __future__ import annotations

import pytest

from evaluation.closed_loop import ClosedLoopMetrics

_S = 1_000_000  # 1 秒的微秒数


def test_full_session_hand_computed() -> None:
    m = ClosedLoopMetrics()
    m.start_autonomous(0)  # AI 开始自主
    m.record_takeover(10 * _S)  # 10s 后人工接管 → +10000ms
    m.start_autonomous(20 * _S)  # 20s 处恢复自主
    m.record_stuck(25 * _S, 5000.0)  # 卡住 5000ms
    m.stop(30 * _S)  # 30s 结束 → 再 +10000ms

    s = m.summary()
    assert s["autonomous_duration_ms"] == pytest.approx(20000.0)
    assert s["takeover_count"] == 1
    # 20000ms = 1/180 小时 → 每小时接管率 180
    assert s["takeover_rate_per_hour"] == pytest.approx(180.0)
    # 窗口 0~30s = 30000ms，stuck 5000ms → 1/6
    assert s["stuck_ratio"] == pytest.approx(1.0 / 6.0)
    assert s["window_duration_ms"] == pytest.approx(30000.0)


def test_stop_finalizes_open_autonomous_segment() -> None:
    m = ClosedLoopMetrics()
    m.start_autonomous(0)
    m.stop(60 * _S)
    s = m.summary()
    assert s["autonomous_duration_ms"] == pytest.approx(60000.0)  # §43 参考线 >=10min 的量纲
    assert s["takeover_count"] == 0
    assert s["takeover_rate_per_hour"] == 0.0
    assert s["stuck_ratio"] == 0.0


def test_takeover_without_autonomous_raises() -> None:
    with pytest.raises(RuntimeError, match="不在自主控制状态"):
        ClosedLoopMetrics().record_takeover(0)


def test_double_start_raises() -> None:
    m = ClosedLoopMetrics()
    m.start_autonomous(0)
    with pytest.raises(RuntimeError, match="重复"):
        m.start_autonomous(1)


def test_negative_stuck_duration_raises() -> None:
    with pytest.raises(ValueError, match="不能为负"):
        ClosedLoopMetrics().record_stuck(0, -1.0)


def test_empty_summary() -> None:
    s = ClosedLoopMetrics().summary()
    assert s["autonomous_duration_ms"] == 0.0
    assert s["takeover_rate_per_hour"] == 0.0
    assert s["stuck_ratio"] == 0.0
