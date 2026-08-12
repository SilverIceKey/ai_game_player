"""推理日志（spec §33 Runtime 日志）。

每次 inference 至少记录 spec §33 规定字段，JSONL 逐行追加。
高频路径（每次推理一条），write() 立即 flush 保证崩溃不丢日志。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from capture.clock import now_us

# spec §33 每次 inference 至少记录的字段
REQUIRED_FIELDS: tuple[str, ...] = (
    "timestamp_us",
    "model_version",
    "observation_id",
    "frame_age_ms",
    "queue_delay_ms",
    "inference_ms",
    "action",
    "action_confidence",
    "mode",
)


class InferenceLogger:
    """推理 JSONL 日志追加器。"""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("a", encoding="utf-8")

    @property
    def path(self) -> Path:
        return self._path

    def write(self, record: dict[str, Any]) -> None:
        """追加一条推理记录；timestamp_us 缺省时打统一时钟。

        除 timestamp_us 外的 §33 必填字段缺失时报错，不允许静默写出
        缺字段日志（下游分析依赖字段完整性）。
        """
        if self._fh.closed:
            raise RuntimeError(f"InferenceLogger 已关闭: {self._path}")
        entry = dict(record)
        entry.setdefault("timestamp_us", now_us())
        missing = [f for f in REQUIRED_FIELDS if f not in entry]
        if missing:
            raise ValueError(f"推理日志缺少必填字段: {missing}（spec §33）")
        self._fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> InferenceLogger:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
