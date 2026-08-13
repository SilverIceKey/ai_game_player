"""model/torch_model.py 单元测试：前向形状 + §18 三阶段冻结（CPU，pretrained=False）。"""
from __future__ import annotations

import torch

from capture.action import BUTTONS
from model.encoding import ACTION_DIM
from model.torch_model import TRAIN_STAGES, VideoActionNet


def _net(**kwargs) -> VideoActionNet:
    params = {
        "history_frames": 2,
        "future_action_steps": 2,
        "camera_bins": 21,
        "hidden_dim": 32,
        "pretrained": False,
    }
    params.update(kwargs)
    return VideoActionNet(**params)


def _batch(net: VideoActionNet, batch: int = 2):
    frames = torch.randn(batch, net.history_frames, 3, 36, 64)
    hist = torch.randn(batch, 3, ACTION_DIM)
    return frames, hist


def test_forward_shapes() -> None:
    net = _net()
    out = net(*_batch(net))
    n = net.future_action_steps
    assert out["move"].shape == (2, n, 2)
    assert out["camera_logits"].shape == (2, n, 2, 21)
    assert out["button_logits"].shape == (2, n, len(BUTTONS))
    # move 经 tanh 限制在 [-1, 1]
    assert out["move"].abs().max() <= 1.0


def test_freeze_backbone_stage() -> None:
    net = _net(train_stage="freeze_backbone")
    assert all(not p.requires_grad for p in net.backbone.parameters())
    assert all(p.requires_grad for p in net.gru.parameters())


def test_unfreeze_last_stage() -> None:
    net = _net(train_stage="unfreeze_last")
    layer4 = net.backbone[-2]
    assert all(p.requires_grad for p in layer4.parameters())
    assert all(not p.requires_grad for p in net.backbone[0].parameters())


def test_full_stage() -> None:
    net = _net(train_stage="full")
    assert all(p.requires_grad for p in net.backbone.parameters())


def test_invalid_stage_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="train_stage"):
        _net(train_stage="yolo")
    assert "freeze_backbone" in TRAIN_STAGES
