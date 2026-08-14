"""model/torch_policy.py 单元测试：checkpoint 往返 + VideoActionPolicy 输出契约 + legacy 拒绝。"""
from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from capture.action import BUTTONS, ActionChunk, ActionRecord, NormalizedAction, SOURCE_AI
from model.policy import PlaceholderPolicy, load_policy
from model.torch_policy import TorchPolicy
from model.torch_model import ARCH_TAG, VideoActionNet
from model.checkpoint import ModelCheckpointMeta, new_checkpoint_meta

_BASE_US = 1_000_000


def _make_checkpoint(tmp_path, memory_slots: int = 2) -> None:
    """落一个随机初始化的 checkpoint（结构与真实训练产物一致）。"""
    net = VideoActionNet(
        history_frames=2, history_actions=2, future_action_steps=2, camera_bins=21,
        d_model=32, num_layers=1, num_heads=4, visual_tokens_per_frame=4,
        memory_slots=memory_slots, pretrained=False,
    )
    meta = new_checkpoint_meta(
        model_version="model-v001",
        dataset_version="dataset-v001",
        code_commit="",
        training_config={
            "arch": ARCH_TAG,
            "history_frames": 2,
            "history_actions": 2,
            "future_action_steps": 2,
            "camera_bins": 21,
            "action_step_ms": 50.0,
            "transformer": {
                "hidden_dim": 32, "num_layers": 1, "num_heads": 4,
                "visual_tokens_per_frame": 4,
            },
            "memory": {"enabled": True, "slots": memory_slots, "update_interval_ms": 100},
        },
    )
    out = tmp_path / "checkpoints" / "model-v001"
    out.mkdir(parents=True)
    import torch

    torch.save(net.state_dict(), out / "model.pt")
    meta.save(out / "meta.json")


def _frames(n: int = 2) -> list[tuple[np.ndarray, int]]:
    return [
        (np.random.rand(36, 64, 3).astype(np.float32), _BASE_US + i * 16_000)
        for i in range(n)
    ]


def _history() -> list[ActionRecord]:
    return [
        ActionRecord(_BASE_US - 50_000, NormalizedAction(move_y=1.0, buttons=frozenset({"dodge"})), SOURCE_AI)
    ]


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

    chunk = policy.predict(_frames(), _history())

    assert isinstance(chunk, ActionChunk)
    assert len(chunk.actions) == 2
    assert chunk.step_ms == 50.0
    assert chunk.model_version == "model-v001"
    assert set(chunk.confidence) == set(BUTTONS)
    for action in chunk.actions:
        assert -1.0 <= action.move_x <= 1.0
        assert -1.0 <= action.camera_x <= 1.0
        assert action.buttons <= set(BUTTONS)
    # spec §16 diagnostics：gate 与 token 计数
    diag = policy.last_diagnostics
    assert 0.0 <= diag["fast_gate"] <= 1.0
    assert 0.0 <= diag["memory_gate"] <= 1.0
    assert diag["visual_tokens"] == 8.0  # 2 帧 × 4 token
    assert diag["memory_updates"] >= 1.0
    for name in ("visual_encode_ms", "transformer_ms", "memory_write_ms", "decode_ms"):
        assert np.isfinite(diag[name]) and diag[name] >= 0.0


def test_visual_token_cache_does_not_reencode_same_frames(tmp_path) -> None:
    _make_checkpoint(tmp_path)
    policy = load_policy(tmp_path / "checkpoints" / "model-v001")
    frames = _frames()
    with patch.object(
        policy._net, "encode_frames", wraps=policy._net.encode_frames
    ) as encode:
        policy.predict(frames, _history())
        policy.predict(frames, _history())
    assert encode.call_count == 1


def test_memory_writer_reuses_cached_latest_visual_tokens(tmp_path) -> None:
    _make_checkpoint(tmp_path)
    policy = load_policy(tmp_path / "checkpoints" / "model-v001")
    with (
        patch.object(policy._net, "encode_frames", wraps=policy._net.encode_frames) as encode,
        patch.object(policy._net, "write_memory", wraps=policy._net.write_memory) as writer,
    ):
        policy.predict(_frames(), _history())
    assert encode.call_count == 1  # 整个窗口一次批量 encode，无 Memory 重编
    assert writer.call_count == 1
    assert writer.call_args.args[0].shape[0] == 1


def test_memory_reset(tmp_path) -> None:
    """Hard Reset（spec §8.3）：清空 memory 槽，计数进 diagnostics。"""
    _make_checkpoint(tmp_path)
    policy = load_policy(tmp_path / "checkpoints" / "model-v001")
    policy.predict(_frames(), _history())
    assert policy.last_diagnostics["memory_slots_filled"] >= 1.0
    policy.reset_memory()
    policy.predict(_frames(), _history())
    assert policy.last_diagnostics["memory_resets"] == 1.0


def test_legacy_checkpoint_rejected(tmp_path) -> None:
    """无 arch 标记的旧 checkpoint（GRU/LSTM 时代）→ legacy/unsupported，不 silent fallback。"""
    _make_checkpoint(tmp_path)
    meta_path = tmp_path / "checkpoints" / "model-v001" / "meta.json"
    import json

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    del meta["training_config"]["arch"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(RuntimeError, match="legacy/unsupported"):
        load_policy(tmp_path / "checkpoints" / "model-v001")


def test_predict_rejects_short_history(tmp_path) -> None:
    _make_checkpoint(tmp_path)
    policy = load_policy(tmp_path / "checkpoints" / "model-v001")
    with pytest.raises(ValueError, match="Video History 不足"):
        policy.predict(_frames(1), [])


def test_missing_weights_file(tmp_path) -> None:
    _make_checkpoint(tmp_path)
    (tmp_path / "checkpoints" / "model-v001" / "model.pt").unlink()
    with pytest.raises(FileNotFoundError, match="model.pt"):
        load_policy(tmp_path / "checkpoints" / "model-v001")


def test_interrupted_training_keeps_completed_epoch_loadable(tmp_path) -> None:
    _make_checkpoint(tmp_path)
    root = tmp_path / "checkpoints" / "model-v001"
    epoch = root / "epochs" / "epoch-001"
    epoch.mkdir(parents=True)
    (root / "model.pt").replace(epoch / "model.pt")
    (root / "meta.json").replace(epoch / "meta.json")
    summary = replace(
        new_checkpoint_meta(
            "model-v001",
            "dataset-v001",
            "",
            ModelCheckpointMeta.load(epoch / "meta.json").training_config,
        ),
        available_epoch_checkpoints=("epochs/epoch-001",),
    )
    summary.save(root / "meta.json")

    assert load_policy(epoch).model_version == "model-v001"
    with pytest.raises(FileNotFoundError, match="尚未生成 final"):
        load_policy(root)
