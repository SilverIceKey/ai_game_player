"""train/dataset.py 单元测试：合成 session → SessionDataset → 张量形状与内容。"""
from __future__ import annotations

import pytest
import torch

from config import ModelConfig, PredictionConfig, Settings
from train.dataset import (
    SessionDataset,
    build_sample_params,
    find_session_dirs,
)
from tests.synth_session import make_synthetic_session


def _settings() -> Settings:
    return Settings(
        game="test",
        model=ModelConfig(
            sample_fps=12.0, history_frames=2, history_actions=2,
            input_width=64, input_height=36,
        ),
        prediction=PredictionConfig(action_step_ms=50.0, future_action_steps=2),
    )


def test_find_session_dirs(tmp_path) -> None:
    make_synthetic_session(tmp_path / "sessions", name="a")
    make_synthetic_session(tmp_path / "sessions", name="b")
    dirs = find_session_dirs(tmp_path / "sessions")
    assert [d.name for d in dirs] == ["a", "b"]
    with pytest.raises(FileNotFoundError):
        find_session_dirs(tmp_path / "nonexistent")
    with pytest.raises(FileNotFoundError, match="manifest"):
        find_session_dirs(tmp_path)


def test_dataset_shapes_and_content(tmp_path) -> None:
    make_synthetic_session(tmp_path / "sessions")
    settings = _settings()
    dataset = SessionDataset(
        find_session_dirs(tmp_path / "sessions"),
        build_sample_params(settings),
        camera_bins=21,
        input_width=64,
        input_height=36,
    )
    assert len(dataset) > 0
    item = dataset[0]
    assert item["frames"].shape == (2, 3, 36, 64)
    assert item["action_hist"].shape == (2, 18)
    assert item["move"].shape == (2, 2)
    assert item["camera_bins"].shape == (2, 2)
    assert item["buttons"].shape == (2, 14)
    # 合成数据恒 move_y=1.0
    assert torch.all(item["move"][:, 1] == 1.0)
    dataset.close()


def test_empty_sessions_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="session_dirs 不能为空"):
        SessionDataset([], build_sample_params(_settings()))
