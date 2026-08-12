"""observability/metrics.py 单元测试：百分位统计与运行计数（spec §32）。"""
from __future__ import annotations

import pytest

from observability.metrics import LatencyStats, RuntimeCounters


def test_latency_percentiles_hand_computed() -> None:
    stats = LatencyStats()
    for v in range(1, 11):  # 1..10ms，共 10 个样本
        stats.add("inference_ms", float(v))
    s = stats.summary()["inference_ms"]
    # 最近秩：idx = round(q/100*(n-1))，n=10
    assert s["p50"] == pytest.approx(5.0)  # round(4.5)=4 → 第 5 个
    assert s["p90"] == pytest.approx(9.0)  # round(8.1)=8 → 第 9 个
    assert s["p95"] == pytest.approx(10.0)  # round(8.55)=9
    assert s["p99"] == pytest.approx(10.0)  # round(8.91)=9
    assert s["max"] == pytest.approx(10.0)


def test_latency_multi_field_isolated() -> None:
    stats = LatencyStats()
    stats.add("capture_ms", 10.0)
    stats.add("inference_ms", 30.0)
    stats.add("inference_ms", 50.0)
    s = stats.summary()
    assert set(s.keys()) == {"capture_ms", "inference_ms"}
    assert s["capture_ms"]["p50"] == 10.0
    # 最近秩：n=2 时 idx = round(0.5*1) = 0 → 较小值
    assert s["inference_ms"]["p50"] == pytest.approx(30.0)
    assert s["inference_ms"]["max"] == pytest.approx(50.0)
    assert stats.count("inference_ms") == 2
    assert stats.count("queue_delay_ms") == 0


def test_latency_empty_field_rejected() -> None:
    with pytest.raises(ValueError, match="不能为空"):
        LatencyStats().add(" ", 1.0)


def test_render_and_write_summary(tmp_path) -> None:
    stats = LatencyStats()
    stats.add("inference_ms", 27.4)
    text = stats.render()
    assert "inference_ms" in text
    assert "p95" in text
    out = stats.write_summary(tmp_path / "sub" / "latency.txt")
    assert out.is_file()
    assert "inference_ms" in out.read_text(encoding="utf-8")


def test_counters_fps_hand_computed() -> None:
    c = RuntimeCounters()
    c.mark_started(0)
    for _ in range(60):
        c.note_frame()
    c.note_frame(dropped=True)
    for _ in range(12):
        c.note_inference()
    assert c.capture_fps(1_000_000) == pytest.approx(60.0)  # 60 帧 / 1s
    assert c.inference_fps(1_000_000) == pytest.approx(12.0)
    assert c.summary() == {
        "frames_captured": 60,
        "frames_dropped": 1,
        "inference_count": 12,
    }


def test_counters_fps_requires_start() -> None:
    c = RuntimeCounters()
    c.note_frame()
    with pytest.raises(RuntimeError, match="mark_started"):
        c.capture_fps(1_000_000)


def test_counters_zero_elapsed_returns_zero() -> None:
    c = RuntimeCounters()
    c.mark_started(1_000_000)
    c.note_frame()
    assert c.capture_fps(1_000_000) == 0.0
