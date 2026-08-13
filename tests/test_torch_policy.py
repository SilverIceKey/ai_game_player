"""model/torch_policy.py 单元测试：checkpoint 往返 + VideoActionPolicy 输出契约。"""
from __future__ import annotations

import numpy as np
import pytest

from capture.action import BUTTONS, ActionChunk, NormalizedAction
from model.policy import PlaceholderPolicy, load_policy
from model.torch_policy import TorchPolicy
from model.torch_model import VideoActionNet
from model.checkpoint import new_checkpoint_meta


def _make_checkpoint(tmp_path) -> None:
    """落一个随机初始化的 checkpoint（结构与真实训练产物一致）。"""
    net = VideoActionNet(
        history_frames=2, future_action_steps=2, camera_bins=21,
        hidden_dim=32, pretrained=False,
    )
    meta = new_checkpoint_meta(
        model_version="model-v001",
        dataset_version="dataset-v001",
        code_commit="",
        training_config={
            "history_frames": 2,
            "future_action_steps": 2,
            "camera_bins": 21,
            "hidden_dim": 32,
            "action_step_ms": 50.0,
        },
    )
    out = tmp_path / "checkpoints" / "model-v001"
    out.mkdir(parents=True)
    import torch

    torch.save(net.state_dict(), out / "model.pt")
    meta.save(out / "meta.json")


def test_load_policy_none_returns_placeholder() -> None:
    assert isinstance(load_policy(None), PlaceholderPolicy)


def test_load_policy_missing_path() -> None:
    with pytest.raises(FileNotFoundError):
        load_policy("nonexistent/path")


def test_checkpoint_roundtrip_and_predict(tmp_path) -> None:
    _make_checkpoint(tmp_path)
    policy = load_policy(tmp_path / "checkpoints" / "model-v001")
    assert isinstance(policy, TorchPolicy)
    assert policy.model_version == "model-v001"

    frames = [np.random.rand(36, 64, 3).astype(np.float32) for _ in range(2)]
    history = [NormalizedAction(move_y=1.0, buttons=frozenset({"dodge"}))]
    chunk = policy.predict(frames, history)

    assert isinstance(chunk, ActionChunk)
    assert len(chunk.actions) == 2
    assert chunk.step_ms == 50.0
    assert chunk.model_version == "model-v001"
    assert set(chunk.confidence) == set(BUTTONS)
    for action in chunk.actions:
        assert -1.0 <= action.move_x <= 1.0
        assert -1.0 <= action.camera_x <= 1.0
        assert action.buttons <= set(BUTTONS)


def test_predict_rejects_short_history(tmp_path) -> None:
    _make_checkpoint(tmp_path)
    policy = load_policy(tmp_path / "checkpoints" / "model-v001")
    with pytest.raises(ValueError, match="Video History 不足"):
        policy.predict([np.zeros((36, 64, 3), dtype=np.float32)], [])


def test_missing_weights_file(tmp_path) -> None:
    _make_checkpoint(tmp_path)
    (tmp_path / "checkpoints" / "model-v001" / "model.pt").unlink()
    with pytest.raises(FileNotFoundError, match="model.pt"):
        load_policy(tmp_path / "checkpoints" / "model-v001")
