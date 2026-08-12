"""observability/logs.py 单元测试：推理 JSONL 写入（spec §33）。"""
from __future__ import annotations

import json

import pytest

from observability.logs import InferenceLogger, REQUIRED_FIELDS


def _record() -> dict:
    return {
        "model_version": "model-v017",
        "observation_id": "obs-001",
        "frame_age_ms": 16.0,
        "queue_delay_ms": 4.0,
        "inference_ms": 27.4,
        "action": {"move_y": 1.0, "dodge": True},
        "action_confidence": {"dodge": 0.9},
        "mode": "AUTOPILOT",
    }


def test_write_appends_jsonl(tmp_path) -> None:
    path = tmp_path / "inference.jsonl"
    logger = InferenceLogger(path)
    logger.write(_record())
    logger.write(_record())
    logger.close()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    entry = json.loads(lines[0])
    for field in REQUIRED_FIELDS:
        assert field in entry
    assert entry["timestamp_us"] > 0  # 缺省时自动打统一时钟
    assert entry["model_version"] == "model-v017"
    assert entry["mode"] == "AUTOPILOT"


def test_write_keeps_explicit_timestamp(tmp_path) -> None:
    path = tmp_path / "inference.jsonl"
    with InferenceLogger(path) as logger:
        rec = _record()
        rec["timestamp_us"] = 87230201120
        logger.write(rec)
    entry = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert entry["timestamp_us"] == 87230201120


def test_missing_required_field_raises(tmp_path) -> None:
    logger = InferenceLogger(tmp_path / "inference.jsonl")
    try:
        with pytest.raises(ValueError, match="缺少必填字段"):
            logger.write({"model_version": "model-v017"})
    finally:
        logger.close()


def test_write_after_close_raises(tmp_path) -> None:
    logger = InferenceLogger(tmp_path / "inference.jsonl")
    logger.close()
    with pytest.raises(RuntimeError, match="已关闭"):
        logger.write(_record())


def test_parent_dir_created(tmp_path) -> None:
    path = tmp_path / "runs" / "session-1" / "inference.jsonl"
    with InferenceLogger(path) as logger:
        logger.write(_record())
    assert path.is_file()
