"""train/trainer.py 单元测试：无 torch 明确报错 + 参数校验（骨架不真训练）。"""
from __future__ import annotations

import importlib.util

import pytest

from config import LossWeights
from train.trainer import Trainer


class FakeBuffer:
    """Replay Buffer 假对象：只实现 trainer 依赖的 sample/size 协议。"""

    def __init__(self, n: int = 0) -> None:
        self._samples = [{"obs": i} for i in range(n)]

    def sample(self, n: int, weights=None) -> list[dict]:
        return self._samples[:n]

    def size(self) -> int:
        return len(self._samples)


def _trainer() -> Trainer:
    return Trainer(
        replay_buffer=FakeBuffer(n=10),
        loss_weights=LossWeights(move=1.0, camera=2.0, button=1.0, temporal=0.5),
        training_config={"lr": 1e-4},
    )


def test_train_candidate_without_torch_raises(tmp_path, monkeypatch) -> None:
    """无论环境是否装了 torch，均模拟缺失，验证错误信息契约。"""
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="PyTorch"):
        _trainer().train_candidate("dataset-v001", "model-v001")


def test_train_candidate_rejects_empty_versions() -> None:
    trainer = _trainer()
    with pytest.raises(ValueError, match="dataset_version"):
        trainer.train_candidate("  ", "model-v001")
    with pytest.raises(ValueError, match="model_version"):
        trainer.train_candidate("dataset-v001", "")


def test_trainer_exposes_loss_weights() -> None:
    assert _trainer().loss_weights.camera == 2.0
