"""model/checkpoint.py 单元测试：元数据 JSON 往返与损坏报错。"""
from __future__ import annotations

import pytest

from model.checkpoint import ModelCheckpointMeta, new_checkpoint_meta


def _meta() -> ModelCheckpointMeta:
    return ModelCheckpointMeta(
        model_version="model-v001",
        dataset_version="dataset-v003",
        code_commit="abc123",
        training_config={"lr": 1e-4, "loss_weights": {"move": 1.0}},
        eval_result={"movement_error": 0.12},
        created_us=123456789,
        epoch=3,
        train_loss={"move": 0.1},
        total_loss=0.2,
        gate={"fast_mean": 0.5},
        available_epoch_checkpoints=("epochs/epoch-001", "epochs/epoch-002"),
        selected_epoch=2,
        selection_reason="last_completed_epoch",
    )


def test_save_load_roundtrip(tmp_path) -> None:
    path = tmp_path / "ckpt" / "meta.json"
    _meta().save(path)
    loaded = ModelCheckpointMeta.load(path)
    assert loaded == _meta()


def test_new_checkpoint_meta_stamps_clock() -> None:
    meta = new_checkpoint_meta("model-v002", "dataset-v001", "deadbeef")
    assert meta.created_us > 0
    assert meta.training_config == {}
    assert meta.eval_result == {}


def test_load_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="不存在"):
        ModelCheckpointMeta.load(tmp_path / "nope.json")


def test_load_corrupted_json_raises(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON 损坏"):
        ModelCheckpointMeta.load(path)


def test_load_missing_required_field_raises(tmp_path) -> None:
    path = tmp_path / "missing.json"
    path.write_text('{"model_version": "model-v001"}', encoding="utf-8")
    with pytest.raises(ValueError, match="缺少必填字段"):
        ModelCheckpointMeta.load(path)


def test_empty_version_rejected() -> None:
    with pytest.raises(ValueError, match="model_version"):
        ModelCheckpointMeta(model_version=" ", dataset_version="dataset-v001", code_commit="")
    with pytest.raises(ValueError, match="dataset_version"):
        ModelCheckpointMeta(model_version="model-v001", dataset_version="", code_commit="")
