"""运行时延迟与计数可观测性（spec §32）。

迁移自旧 core/trace.py 的 TickTrace（P50/P95/max），扩展为：
- LatencyStats：按字段累计延迟样本，输出 P50/P90/P95/P99/MAX
  （最近秩百分位；spec §32：不能只看平均延迟）；
- RuntimeCounters：Capture FPS / Dropped Frames / Inference 计数等
  （spec §32 同时要求的运行计数；GPU Utilization / VRAM / Temperature
  依赖 torch/驱动，留待训练链路落地后补充）。
"""
from __future__ import annotations

from pathlib import Path


def _percentile(sorted_vals: list[float], q: float) -> float:
    """最近秩百分位（sorted_vals 非空）。"""
    idx = min(len(sorted_vals) - 1, max(0, round(q / 100.0 * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


class LatencyStats:
    """按字段分组的延迟百分位统计（毫秒）。"""

    _QS = (50, 90, 95, 99)

    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = {}

    def add(self, field: str, value_ms: float) -> None:
        """记录 field 的一次延迟样本（毫秒）。"""
        if not field.strip():
            raise ValueError("field 不能为空")
        self._samples.setdefault(field, []).append(float(value_ms))

    def count(self, field: str) -> int:
        return len(self._samples.get(field, ()))

    def summary(self) -> dict[str, dict[str, float]]:
        """{field: {p50, p90, p95, p99, max}}；无样本的字段不出现于结果。"""
        out: dict[str, dict[str, float]] = {}
        for field, values in self._samples.items():
            if not values:
                continue
            vals = sorted(values)
            stats = {f"p{q}": _percentile(vals, q) for q in self._QS}
            stats["max"] = vals[-1]
            out[field] = stats
        return out

    def render(self) -> str:
        lines = [f"{'field':<20}{'p50':>10}{'p90':>10}{'p95':>10}{'p99':>10}{'max':>10}"]
        for field, stats in self.summary().items():
            lines.append(
                f"{field:<20}{stats['p50']:>9.1f} {stats['p90']:>9.1f} "
                f"{stats['p95']:>9.1f} {stats['p99']:>9.1f} {stats['max']:>9.1f}"
            )
        return "\n".join(lines)

    def write_summary(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.render() + "\n", encoding="utf-8")
        return out


class RuntimeCounters:
    """运行计数器（spec §32：Capture FPS / Dropped Frames / Inference FPS 等）。

    时间戳由调用方以统一时钟传入（capture.clock.now_us()，spec §11），
    本类不自行取时，保持纯逻辑可测。
    """

    def __init__(self) -> None:
        self.frames_captured = 0
        self.frames_dropped = 0
        self.inference_count = 0
        self._started_us: int | None = None

    def mark_started(self, now_us: int) -> None:
        """标记计数窗口起点（进程/会话启动时调用一次）。"""
        self._started_us = now_us

    def note_frame(self, dropped: bool = False) -> None:
        """记录一帧采集结果。"""
        if dropped:
            self.frames_dropped += 1
        else:
            self.frames_captured += 1

    def note_inference(self) -> None:
        """记录一次推理完成。"""
        self.inference_count += 1

    def _fps(self, count: int, now_us: int) -> float:
        if self._started_us is None:
            raise RuntimeError("未调用 mark_started()，无法计算 FPS")
        elapsed_s = (now_us - self._started_us) / 1_000_000.0
        return count / elapsed_s if elapsed_s > 0 else 0.0

    def capture_fps(self, now_us: int) -> float:
        """截至 now_us 的平均采集帧率。"""
        return self._fps(self.frames_captured, now_us)

    def inference_fps(self, now_us: int) -> float:
        """截至 now_us 的平均推理频率。"""
        return self._fps(self.inference_count, now_us)

    def summary(self) -> dict[str, int]:
        return {
            "frames_captured": self.frames_captured,
            "frames_dropped": self.frames_dropped,
            "inference_count": self.inference_count,
        }
