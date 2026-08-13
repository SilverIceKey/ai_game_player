"""model/policy.py 单元测试：占位/随机 Policy 输出契约与 load_policy 行为。"""
from __future__ import annotations

import pytest

from capture.action import ActionChunk, NormalizedAction
from model.policy import (
    PlaceholderPolicy,
    RandomPolicy,
    VideoActionPolicy,
    load_policy,
)


def test_placeholder_returns_neutral_chunk() -> None:
    policy = PlaceholderPolicy()
    chunk = policy.predict(frames=[], action_history=[])
    assert isinstance(chunk, ActionChunk)
    assert len(chunk.actions) == 4
    assert chunk.step_ms == 50.0
    assert chunk.model_version == "placeholder"
    assert chunk.created_us > 0
    for action in chunk.actions:
        assert action.is_neutral()


def test_placeholder_custom_steps() -> None:
    policy = PlaceholderPolicy(future_action_steps=8, step_ms=25.0)
    chunk = policy.predict([], [])
    assert len(chunk.actions) == 8
    assert chunk.step_ms == 25.0


def test_placeholder_rejects_invalid_params() -> None:
    with pytest.raises(ValueError):
        PlaceholderPolicy(future_action_steps=0)
    with pytest.raises(ValueError):
        PlaceholderPolicy(step_ms=0)


def test_random_policy_deterministic_with_seed() -> None:
    a = RandomPolicy(seed=42).predict([], [])
    b = RandomPolicy(seed=42).predict([], [])
    assert a.actions == b.actions
    assert len(a.actions) == 4
    assert a.model_version == "random"


def test_random_policy_different_seeds_differ() -> None:
    a = RandomPolicy(seed=1).predict([], [])
    b = RandomPolicy(seed=2).predict([], [])
    assert a.actions != b.actions


def test_random_policy_actions_in_range() -> None:
    chunk = RandomPolicy(seed=7).predict([], [])
    for action in chunk.actions:
        assert -1.0 <= action.move_x <= 1.0
        assert -1.0 <= action.move_y <= 1.0
        assert -1.0 <= action.camera_x <= 1.0
        assert -1.0 <= action.camera_y <= 1.0
        assert isinstance(action, NormalizedAction)


def test_policies_satisfy_protocol() -> None:
    assert isinstance(PlaceholderPolicy(), VideoActionPolicy)
    assert isinstance(RandomPolicy(seed=0), VideoActionPolicy)


def test_load_policy_without_checkpoint_returns_placeholder() -> None:
    policy = load_policy(None)
    assert isinstance(policy, PlaceholderPolicy)


def test_load_policy_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="checkpoint 不存在"):
        load_policy(tmp_path / "nonexistent.pt")


def test_load_policy_without_torch_raises_runtime_error(tmp_path, monkeypatch) -> None:
    """环境无 torch 时必须给明确报错与指引（无论环境是否真有 torch，均模拟缺失）。"""
    import importlib.util

    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"fake")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="PyTorch"):
        load_policy(ckpt)
