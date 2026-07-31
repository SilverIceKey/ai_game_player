"""M3 Trace 延迟统计测试。"""
import pytest

from core.trace import TickTrace


def test_trace_summary_percentiles(tmp_path):
    trace = TickTrace()
    for i in range(100):
        trace.add(10.0 + i, 1.0, 2.0)  # perceive 10..109
    summary = trace.summary()
    assert summary["perceive_ms"]["p50"] == pytest.approx(60.0)   # sorted[50]
    assert summary["perceive_ms"]["p95"] == pytest.approx(104.0)  # sorted[94]
    assert summary["perceive_ms"]["max"] == pytest.approx(109.0)
    assert summary["decide_ms"]["p50"] == pytest.approx(1.0)
    assert summary["total_ms"]["max"] == pytest.approx(112.0)

    out = trace.write_summary(tmp_path / "trace_summary.txt")
    text = out.read_text(encoding="utf-8")
    assert "ticks=100" in text
    assert "perceive_ms" in text and "p95" in text


def test_trace_empty():
    trace = TickTrace()
    assert trace.count == 0
    assert trace.summary()["perceive_ms"] == {"p50": 0.0, "p95": 0.0, "max": 0.0}
    assert "ticks=0" in trace.render()
