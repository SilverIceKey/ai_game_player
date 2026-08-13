"""train/losses.py 单元测试：§23 组合 loss + §24 pos_weight。"""
from __future__ import annotations

import pytest
import torch

from config import LossWeights
from train.losses import compute_button_pos_weight, compute_loss


def _make(B: int = 2, n: int = 4, bins: int = 21, buttons: int = 14):
    outputs = {
        "move": torch.rand(B, n, 2) * 2 - 1,
        "camera_logits": torch.randn(B, n, 2, bins),
        "button_logits": torch.randn(B, n, buttons),
    }
    targets = {
        "move": torch.rand(B, n, 2) * 2 - 1,
        "camera_bins": torch.randint(0, bins, (B, n, 2)),
        "buttons": (torch.rand(B, n, buttons) > 0.8).float(),
    }
    return outputs, targets


def test_loss_parts_and_weights() -> None:
    outputs, targets = _make()
    weights = LossWeights(move=1.0, camera=2.0, button=1.0, temporal=0.5)
    total, parts = compute_loss(outputs, targets, weights)
    assert set(parts) == {"move", "camera", "button", "temporal"}
    expected = parts["move"] + 2.0 * parts["camera"] + parts["button"] + 0.5 * parts["temporal"]
    assert float(total) == pytest.approx(expected, rel=1e-5)


def test_loss_backward() -> None:
    outputs, targets = _make()
    for value in outputs.values():
        value.requires_grad_(True)
    total, _ = compute_loss(outputs, targets, LossWeights())
    total.backward()
    assert outputs["move"].grad is not None


def test_single_step_chunk_temporal_zero() -> None:
    outputs, targets = _make(n=1)
    _, parts = compute_loss(outputs, targets, LossWeights())
    assert parts["temporal"] == 0.0


def test_pos_weight_imbalance() -> None:
    # 按钮 0 高频（全 1），按钮 1 稀有（1/100）
    buttons = torch.zeros(10, 10, 14)
    buttons[:, :, 0] = 1.0
    buttons[0, 0, 1] = 1.0
    pw = compute_button_pos_weight(buttons)
    assert pw.shape == (14,)
    assert pw[1] > pw[0]  # 稀有按钮权重大
    assert pw[0] == pytest.approx(0.0)  # 全 1 → neg=0
    # 全 0 按钮：pos=0 → clamp 不爆
    assert torch.isfinite(pw).all()
