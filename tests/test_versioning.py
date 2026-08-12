"""dataset/versioning.py 单元测试：版本号递增与元信息 JSON 往返。"""
from __future__ import annotations

import pytest

from dataset.versioning import (
    DatasetVersionMeta,
    next_dataset_version,
    parse_dataset_version,
)


def test_next_version_empty():
    assert next_dataset_version([]) == "dataset-v001"


def test_next_version_increments_max():
    assert next_dataset_version(["dataset-v001"]) == "dataset-v002"
    assert next_dataset_version(["dataset-v001", "dataset-v003", "dataset-v002"]) == "dataset-v004"
    assert next_dataset_version(["dataset-v009"]) == "dataset-v010"


def test_parse_version():
    assert parse_dataset_version("dataset-v001") == 1
    assert parse_dataset_version("dataset-v1234") == 1234


def test_invalid_version_raises():
    for bad in ("v001", "dataset-v1", "dataset-v01", "dataset-vabc", "dataset_v001", ""):
        with pytest.raises(ValueError, match="非法数据集版本号"):
            parse_dataset_version(bad)
    with pytest.raises(ValueError, match="非法数据集版本号"):
        next_dataset_version(["dataset-v001", "bogus"])


def test_meta_save_load_roundtrip(tmp_path):
    meta = DatasetVersionMeta(
        version="dataset-v002",
        created_us=87_230_199_210,
        session_ids=["20260812_001", "20260812_002"],
        sample_counts={"historical": 1200, "recent": 300, "correction": 80, "rare": 12},
        description="首次合并 correction 数据",
    )
    path = tmp_path / "datasets" / "dataset-v002.json"
    meta.save(path)

    loaded = DatasetVersionMeta.load(path)
    assert loaded == meta
    assert loaded.sample_counts["correction"] == 80


def test_meta_invalid_version_rejected():
    with pytest.raises(ValueError, match="非法数据集版本号"):
        DatasetVersionMeta(version="v2", created_us=0)


def test_meta_load_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        DatasetVersionMeta.load(tmp_path / "nope.json")


def test_meta_load_corrupt_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.json"):
        DatasetVersionMeta.load(path)


def test_meta_load_missing_required_field(tmp_path):
    path = tmp_path / "missing.json"
    path.write_text('{"version": "dataset-v001"}', encoding="utf-8")
    with pytest.raises(ValueError, match="缺少必填字段"):
        DatasetVersionMeta.load(path)
