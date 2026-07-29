"""JsonlRecorder：StepRecord 逐行落 JSONL + 关键帧截图落盘。

回放产物（runs/<timestamp>/session.jsonl + frames/）是 LLM 离线复盘
（llm/review，M2）的唯一输入。
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from core.contracts import Action, Suggestion
from core.recorder.base import StepRecord


class JsonlRecorder:
    """实现 core.recorder.base.Recorder 契约。"""

    def __init__(self, run_dir: str | Path, filename: str = "session.jsonl"):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir = self.run_dir / "frames"
        self.frames_dir.mkdir(exist_ok=True)
        self._path = self.run_dir / filename
        self._fp = self._path.open("a", encoding="utf-8")

    def record(self, step: StepRecord) -> None:
        self._fp.write(json.dumps(self._serialize(step), ensure_ascii=False) + "\n")
        self._fp.flush()

    def save_frame(self, frame: np.ndarray, tag: str) -> Path:
        """落关键帧截图（状态转移/首次接敌），返回文件路径。"""
        import cv2

        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in tag)
        path = self.frames_dir / f"{safe}.png"
        if not cv2.imwrite(str(path), frame):
            raise OSError(f"关键帧写入失败: {path}")
        return path

    def export(self) -> Path:
        """导出本次会话的回放文件路径。"""
        self._fp.flush()
        return self._path

    def close(self) -> None:
        if not self._fp.closed:
            self._fp.close()

    @staticmethod
    def _serialize(step: StepRecord) -> dict:
        if isinstance(step.output, Suggestion):
            output = {
                "type": "suggestion",
                "action": asdict(step.output.action),
                "reason": step.output.reason,
                "confidence": step.output.confidence,
            }
        elif isinstance(step.output, Action):
            output = {"type": "action", **asdict(step.output)}
        else:
            raise TypeError(f"不支持的决策输出类型: {type(step.output).__name__}")
        return {
            "timestamp": step.timestamp,
            "state": asdict(step.state),
            "output": output,
            "result": step.result,
            "extra": step.extra,
        }
