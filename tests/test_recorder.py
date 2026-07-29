"""JsonlRecorder 落盘与导出测试。"""
import json

import numpy as np

from core.contracts import Action, GameState, Suggestion
from core.recorder.base import StepRecord
from core.recorder.jsonl import JsonlRecorder


def _step(ts: float, output) -> StepRecord:
    return StepRecord(
        timestamp=ts,
        state=GameState(timestamp=ts, scene="combat", raw={"hp_ratio": 0.5}),
        output=output,
        result="ok",
    )


def test_record_and_export(tmp_path):
    rec = JsonlRecorder(tmp_path / "run1")
    rec.record(_step(1.0, Action("light_attack", {"combo": 1})))
    rec.record(_step(2.0, Suggestion(action=Action("dodge"), reason="低血", confidence=0.7)))
    path = rec.export()
    rec.close()

    assert path == tmp_path / "run1" / "session.jsonl"
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    assert first["output"]["type"] == "action"
    assert first["output"]["name"] == "light_attack"
    assert first["output"]["params"] == {"combo": 1}
    assert first["state"]["scene"] == "combat"
    assert first["state"]["raw"]["hp_ratio"] == 0.5
    assert first["result"] == "ok"

    second = json.loads(lines[1])
    assert second["output"]["type"] == "suggestion"
    assert second["output"]["action"]["name"] == "dodge"
    assert second["output"]["reason"] == "低血"
    assert second["output"]["confidence"] == 0.7


def test_export_returns_path_under_run_dir(tmp_path):
    rec = JsonlRecorder(tmp_path / "run2")
    rec.record(_step(1.0, Action("idle")))
    path = rec.export()
    rec.close()
    assert path.parent.name == "run2"
    assert path.exists()


def test_save_frame(tmp_path):
    rec = JsonlRecorder(tmp_path / "run3")
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    path = rec.save_frame(frame, "000042_COMBAT")
    rec.close()
    assert path.exists()
    assert path.parent.name == "frames"
    assert path.suffix == ".png"


def test_save_frame_sanitizes_tag(tmp_path):
    rec = JsonlRecorder(tmp_path / "run4")
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    path = rec.save_frame(frame, "a/b:c")
    rec.close()
    assert "/" not in path.name and ":" not in path.name
    assert path.exists()
