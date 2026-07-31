"""TickTrace：每 tick 感知/决策/执行延迟统计（M3 计划 2.6）。

session 结束写 runs/<ts>/trace_summary.txt（P50/P95/max）并打日志。
"""
from __future__ import annotations

from pathlib import Path

_FIELDS = ("perceive_ms", "decide_ms", "execute_ms", "total_ms")


def _percentile(sorted_vals: list[float], q: float) -> float:
    """最近秩百分位（sorted_vals 非空）。"""
    idx = min(len(sorted_vals) - 1, max(0, round(q / 100.0 * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


class TickTrace:
    def __init__(self):
        self._records: list[tuple[float, float, float]] = []

    def add(self, perceive_ms: float, decide_ms: float, execute_ms: float) -> None:
        self._records.append((float(perceive_ms), float(decide_ms), float(execute_ms)))

    @property
    def count(self) -> int:
        return len(self._records)

    def summary(self) -> dict[str, dict[str, float]]:
        """{字段: {p50, p95, max}}；无记录时全 0。"""
        out: dict[str, dict[str, float]] = {}
        columns = [
            [r[0] for r in self._records],
            [r[1] for r in self._records],
            [r[2] for r in self._records],
            [sum(r) for r in self._records],
        ]
        for name, values in zip(_FIELDS, columns, strict=True):
            if not values:
                out[name] = {"p50": 0.0, "p95": 0.0, "max": 0.0}
                continue
            values = sorted(values)
            out[name] = {
                "p50": _percentile(values, 50),
                "p95": _percentile(values, 95),
                "max": values[-1],
            }
        return out

    def render(self) -> str:
        lines = [f"tick trace (ticks={self.count})"]
        lines.append(f"{'segment':<14}{'p50':>10}{'p95':>10}{'max':>10}")
        for name, stats in self.summary().items():
            lines.append(
                f"{name:<14}{stats['p50']:>9.1f} {stats['p95']:>9.1f} {stats['max']:>9.1f}"
            )
        return "\n".join(lines)

    def write_summary(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.render() + "\n", encoding="utf-8")
        return out
