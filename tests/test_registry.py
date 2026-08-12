"""train/registry.py 单元测试：状态流转（spec §7）与 JSON 持久化。"""
from __future__ import annotations

import pytest

from model.checkpoint import ModelCheckpointMeta
from train.registry import ModelRegistry


def _meta(version: str) -> ModelCheckpointMeta:
    return ModelCheckpointMeta(
        model_version=version,
        dataset_version="dataset-v001",
        code_commit="abc",
        created_us=1,
    )


def _registry(tmp_path) -> ModelRegistry:
    return ModelRegistry(tmp_path / "registry.json")


def test_register_and_no_active(tmp_path) -> None:
    r = _registry(tmp_path)
    r.register_candidate(_meta("model-v001"))
    assert r.status("model-v001") == "candidate"
    assert r.active_model() is None


def test_promote_requires_evaluation(tmp_path) -> None:
    r = _registry(tmp_path)
    r.register_candidate(_meta("model-v001"))
    with pytest.raises(ValueError, match="evaluated"):
        r.promote("model-v001")


def test_full_lifecycle(tmp_path) -> None:
    r = _registry(tmp_path)
    r.register_candidate(_meta("model-v001"))
    r.record_evaluation("model-v001", {"movement_error": 0.1})
    assert r.status("model-v001") == "evaluated"
    r.promote("model-v001")
    assert r.status("model-v001") == "active"
    assert r.active_model().model_version == "model-v001"


def test_promote_demotes_previous_active(tmp_path) -> None:
    r = _registry(tmp_path)
    for v in ("model-v001", "model-v002"):
        r.register_candidate(_meta(v))
        r.record_evaluation(v, {"movement_error": 0.1})
    r.promote("model-v001")
    r.promote("model-v002")
    assert r.status("model-v001") == "evaluated"
    assert r.active_model().model_version == "model-v002"


def test_reject_candidate(tmp_path) -> None:
    r = _registry(tmp_path)
    r.register_candidate(_meta("model-v001"))
    r.reject("model-v001")
    assert r.status("model-v001") == "rejected"


def test_reject_active_raises(tmp_path) -> None:
    r = _registry(tmp_path)
    r.register_candidate(_meta("model-v001"))
    r.record_evaluation("model-v001", {})
    r.promote("model-v001")
    with pytest.raises(ValueError, match="active"):
        r.reject("model-v001")


def test_evaluation_on_rejected_raises(tmp_path) -> None:
    r = _registry(tmp_path)
    r.register_candidate(_meta("model-v001"))
    r.reject("model-v001")
    with pytest.raises(ValueError, match="rejected"):
        r.record_evaluation("model-v001", {})


def test_duplicate_register_raises(tmp_path) -> None:
    r = _registry(tmp_path)
    r.register_candidate(_meta("model-v001"))
    with pytest.raises(ValueError, match="已注册"):
        r.register_candidate(_meta("model-v001"))


def test_unregistered_model_raises(tmp_path) -> None:
    r = _registry(tmp_path)
    with pytest.raises(KeyError, match="未注册"):
        r.status("model-v999")


def test_persistence_roundtrip(tmp_path) -> None:
    path = tmp_path / "registry.json"
    r = ModelRegistry(path)
    r.register_candidate(_meta("model-v001"))
    r.record_evaluation("model-v001", {"movement_error": 0.1})
    r.promote("model-v001")

    loaded = ModelRegistry.load(path)
    assert loaded.active_model().model_version == "model-v001"
    assert loaded.active_model().eval_result == {"movement_error": 0.1}
    assert loaded.list_models() == {"model-v001": "active"}


def test_load_missing_file_returns_empty(tmp_path) -> None:
    loaded = ModelRegistry.load(tmp_path / "nope.json")
    assert loaded.active_model() is None
    assert loaded.list_models() == {}


def test_load_corrupted_file_raises(tmp_path) -> None:
    path = tmp_path / "registry.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON 损坏"):
        ModelRegistry.load(path)
