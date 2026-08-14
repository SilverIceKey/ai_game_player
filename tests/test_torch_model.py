"""model/torch_model.py 单元测试：Token Transformer 前向形状 + §18 三阶段冻结（CPU，pretrained=False）。"""
from __future__ import annotations

import pytest
import torch

from capture.action import BUTTONS
from model.encoding import ACTION_DIM
from model.torch_model import (
    ARCH_TAG,
    PAD_AGE_S,
    TRAIN_STAGES,
    VideoActionNet,
)


def _net(**kwargs) -> VideoActionNet:
    params = {
        "history_frames": 2,
        "history_actions": 2,
        "future_action_steps": 2,
        "camera_bins": 21,
        "d_model": 32,
        "num_layers": 2,
        "num_heads": 4,
        "visual_tokens_per_frame": 4,
        "pretrained": False,
    }
    params.update(kwargs)
    return VideoActionNet(**params)


def _batch(net: VideoActionNet, batch: int = 2) -> dict[str, torch.Tensor]:
    b = {
        "frames": torch.randn(batch, net.history_frames, 3, 36, 64),
        "frame_ages": torch.rand(batch, net.history_frames) * 2,
        "action_hist": torch.randn(batch, net.history_actions, ACTION_DIM),
        "action_ages": torch.rand(batch, net.history_actions),
    }
    if net.memory_slots > 0:
        b["memory_frames"] = torch.randn(batch, net.memory_slots, 3, 36, 64)
        b["memory_ages"] = torch.rand(batch, net.memory_slots) * 10
    return b


def _assert_heads(net: VideoActionNet, out: dict[str, torch.Tensor], batch: int = 2) -> None:
    n = net.future_action_steps
    assert out["move"].shape == (batch, n, 2)
    assert out["camera_logits"].shape == (batch, n, 2, 21)
    assert out["button_logits"].shape == (batch, n, len(BUTTONS))
    assert out["gates"].shape == (batch, 2)
    assert ((out["gates"] >= 0) & (out["gates"] <= 1)).all()
    # move 经 tanh 限制在 [-1, 1]
    assert out["move"].abs().max() <= 1.0


def test_forward_shapes() -> None:
    net = _net()
    _assert_heads(net, net(**_batch(net)))


def test_forward_with_memory() -> None:
    net = _net(memory_slots=2)
    _assert_heads(net, net(**_batch(net)))


def test_forward_tokens_matches_full_frame_forward() -> None:
    net = _net(dropout=0.0).eval()
    batch = _batch(net)
    frames = batch["frames"]
    with torch.inference_mode():
        expected = net(**batch)
        b, k = frames.shape[:2]
        tokens = net.encode_frames(frames.reshape(b * k, *frames.shape[2:])).reshape(
            b, k, net.visual_tokens_per_frame, net.d_model
        )
        cached = net.forward_tokens(
            tokens,
            batch["frame_ages"],
            batch["action_hist"],
            batch["action_ages"],
        )
    for name in expected:
        assert torch.isfinite(expected[name]).all()
        assert torch.allclose(expected[name], cached[name], atol=1e-6, rtol=1e-5)


def test_memory_required_when_enabled() -> None:
    net = _net(memory_slots=2)
    b = _batch(net)
    del b["memory_frames"], b["memory_ages"]
    with pytest.raises(ValueError, match="memory"):
        net(**b)


def test_pad_slots_finite() -> None:
    # 空槽（内容置零 + 年龄 PAD_AGE_S）不应产生 NaN/Inf
    net = _net()
    b = _batch(net)
    b["action_hist"].zero_()
    b["action_ages"].fill_(PAD_AGE_S)
    out = net(**b)
    for v in out.values():
        assert torch.isfinite(v).all()


def test_older_action_changes_output() -> None:
    # age bias 生效的粗检：同一 action 内容，年龄越大对输出影响越弱方向性不验，
    # 只验证年龄确实参与计算（改年龄输出必变）
    net = _net()
    b = _batch(net)
    out_new = net(**b)["move"]
    b["action_ages"] = b["action_ages"] + 5.0
    out_old = net(**b)["move"]
    assert not torch.allclose(out_new, out_old)


def test_invalid_visual_tokens_rejected() -> None:
    with pytest.raises(ValueError, match="visual_tokens_per_frame"):
        _net(visual_tokens_per_frame=5)


def test_future_latent_head_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        _net(future_latent_head=True)


def test_freeze_backbone_stage() -> None:
    net = _net(train_stage="freeze_backbone")
    assert all(not p.requires_grad for p in net.backbone.parameters())
    assert all(p.requires_grad for p in net.encoder.parameters())


def test_unfreeze_last_stage() -> None:
    net = _net(train_stage="unfreeze_last")
    layer4 = net.backbone[-1]  # 去 fc/avgpool 后最后一层即 layer4
    assert all(p.requires_grad for p in layer4.parameters())
    assert all(not p.requires_grad for p in net.backbone[0].parameters())


def test_full_stage() -> None:
    net = _net(train_stage="full")
    assert all(p.requires_grad for p in net.backbone.parameters())


def test_invalid_stage_rejected() -> None:
    with pytest.raises(ValueError, match="train_stage"):
        _net(train_stage="yolo")
    assert "freeze_backbone" in TRAIN_STAGES
    assert ARCH_TAG == "token_transformer_v1"
