"""train/trainer.py 单元测试：无 torch 报错契约 + 合成数据小训练（CPU）。

合成数据端到端小训练即 spec §42 Phase 1 的最小验证：loss 必须下降，
checkpoint（model.pt + meta.json）落盘且元数据完整（§29）。
"""
from __future__ import annotations

import importlib.util

import pytest
import torch

from config import (
    LossWeights,
    ModelConfig,
    PredictionConfig,
    Settings,
    TrainingConfig,
)
from train.dataset import SessionDataset, build_sample_params, find_session_dirs
from train.trainer import Trainer
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


def _trainer(training: TrainingConfig | None = None) -> Trainer:
    settings = _settings()
    return Trainer(
        loss_weights=LossWeights(),
        training=training or TrainingConfig(epochs=2, batch_size=8),
        model_config=settings.model,
        prediction=settings.prediction,
        device=torch.device("cpu"),
        pretrained=False,  # 测试不下载 ImageNet 权重
    )


def _dataset(tmp_path) -> SessionDataset:
    make_synthetic_session(tmp_path / "sessions")
    settings = _settings()
    return SessionDataset(
        find_session_dirs(tmp_path / "sessions"),
        build_sample_params(settings),
        camera_bins=21,
        input_width=64,
        input_height=36,
    )


def test_without_torch_raises(tmp_path, monkeypatch) -> None:
    """无 torch 环境构造即明确报错（monkeypatch 模拟缺失，与是否真装无关）。"""
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="PyTorch"):
        _trainer()


def test_train_candidate_tiny_overfit(tmp_path, capsys) -> None:
    """合成数据 2 epoch：loss 下降 + checkpoint 落盘 + meta 五要素齐全。"""
    dataset = _dataset(tmp_path)
    meta = _trainer().train_candidate(
        dataset,
        dataset_version="dataset-v001",
        model_version="model-v001",
        code_commit="deadbeef",
        checkpoints_dir=tmp_path / "checkpoints",
    )
    dataset.close()

    out_log = capsys.readouterr().out
    assert "[train] 开始训练" in out_log
    assert "batch " in out_log and "samples/s" in out_log  # batch 级进度日志
    assert "epoch 2/2 done" in out_log

    history = meta.eval_result["loss_history"]
    assert history[-1]["total"] < history[0]["total"]  # Phase 1：明显拟合趋势
    assert meta.dataset_version == "dataset-v001"
    assert meta.code_commit == "deadbeef"
    assert meta.training_config["history_frames"] == 2
    assert meta.training_config["future_action_steps"] == 2
    assert "movement_error" in meta.eval_result

    out = tmp_path / "checkpoints" / "model-v001"
    assert (out / "model.pt").is_file()
    assert (out / "meta.json").is_file()


def test_rejects_empty_versions(tmp_path) -> None:
    dataset = _dataset(tmp_path)
    trainer = _trainer(TrainingConfig(epochs=1, batch_size=8))
    with pytest.raises(ValueError, match="dataset_version"):
        trainer.train_candidate(dataset, " ", "model-v001", checkpoints_dir=tmp_path)
    with pytest.raises(ValueError, match="model_version"):
        trainer.train_candidate(dataset, "dataset-v001", "", checkpoints_dir=tmp_path)
    dataset.close()
