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


def test_frame_cache_dedupes_and_reuses(tmp_path) -> None:
    """帧缓存：相邻样本重叠的帧只解码一次；两次取同一样本结果一致（走缓存）。"""
    make_synthetic_session(tmp_path / "sessions")
    settings = _settings()
    dataset = SessionDataset(
        find_session_dirs(tmp_path / "sessions"),
        build_sample_params(settings),
        camera_bins=21,
        input_width=64,
        input_height=36,
    )
    # 样本数 × history_frames 远大于去重后的唯一帧数（相邻样本窗口高度重叠）
    unique = len(dataset._frame_cache)
    assert unique < len(dataset) * settings.model.history_frames
    first = dataset[0]["frames"]
    second = dataset[0]["frames"]
    assert torch.equal(first, second)
    # uint8 resize 缓存：值与直接 normalize 原帧等价（0 通道 = frame_id % 256 的 B 通道）
    assert first.shape == (2, 3, 36, 64)
    dataset.close()


def test_frame_cache_falls_back_to_disk_memmap(tmp_path, monkeypatch) -> None:
    """可用内存不足阈值时帧缓存落磁盘 memmap，取数正常，close 清理文件。"""
    import train.dataset as dataset_mod

    monkeypatch.setattr(dataset_mod, "_available_ram_bytes", lambda: 1)  # 1 字节可用 → 强制磁盘
    make_synthetic_session(tmp_path / "sessions")
    settings = _settings()
    dataset = SessionDataset(
        find_session_dirs(tmp_path / "sessions"),
        build_sample_params(settings),
        camera_bins=21,
        input_width=64,
        input_height=36,
    )
    mmap_path = tmp_path / "sessions" / ".frame_cache.npy"
    assert dataset._mmap is not None and mmap_path.is_file()
    item = dataset[0]
    assert item["frames"].shape == (2, 3, 36, 64)
    dataset.close()
    assert not mmap_path.exists()
