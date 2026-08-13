"""Behavior Cloning 组合 Loss（spec §23）+ 类别不平衡处理（§24）。

    L = L_move + λc·L_camera + λb·L_button + λt·L_temporal

- L_move：Movement Head MSE（§19.1 regression）
- L_camera：Camera Head 离散 bin 交叉熵（§19.2 discretized distribution）
- L_button：Button Head multi-label BCEWithLogits（§19.3），pos_weight
  按 §24 自动统计（闪避/喝药等低频高价值动作加权）
- L_temporal：chunk 内相邻步预测的平滑惩罚（减弱动作抖动，§15 目的之一）

权重全部来自 config.LossWeights（§23：必须配置化并记录实验）。

本模块顶层 import torch：只在训练路径上延迟导入。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from config import LossWeights


def compute_button_pos_weight(target_buttons: torch.Tensor) -> torch.Tensor:
    """按 §24 统计按钮 pos_weight = neg/pos（clamp 防爆）。"""
    pos = target_buttons.sum(dim=(0, 1))  # (14,)
    neg = target_buttons.shape[0] * target_buttons.shape[1] - pos
    return (neg / pos.clamp(min=1.0)).clamp(max=50.0)


def compute_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    weights: LossWeights,
    pos_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """组合 loss。targets: move (B,n,2) / camera_bins (B,n,2) long / buttons (B,n,14)。

    返回 (total, 分项数值 dict)——分项进训练日志，定位哪一路学不动（spec §17 排查顺序）。
    """
    l_move = F.mse_loss(outputs["move"], targets["move"])

    logits = outputs["camera_logits"]  # (B,n,2,bins)
    l_camera = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets["camera_bins"].reshape(-1),
    )

    l_button = F.binary_cross_entropy_with_logits(
        outputs["button_logits"], targets["buttons"], pos_weight=pos_weight
    )

    # 相邻步平滑：move 与按钮概率的步间差（第一步无差分，步数>1 才有意义）
    if outputs["move"].shape[1] > 1:
        move_diff = outputs["move"][:, 1:] - outputs["move"][:, :-1]
        btn_diff = (
            torch.sigmoid(outputs["button_logits"][:, 1:])
            - torch.sigmoid(outputs["button_logits"][:, :-1])
        )
        l_temporal = move_diff.pow(2).mean() + btn_diff.pow(2).mean()
    else:
        l_temporal = torch.zeros((), device=outputs["move"].device)

    total = (
        l_move
        + weights.camera * l_camera
        + weights.button * l_button
        + weights.temporal * l_temporal
    )
    parts = {
        "move": float(l_move.detach()),
        "camera": float(l_camera.detach()),
        "button": float(l_button.detach()),
        "temporal": float(l_temporal.detach()),
    }
    return total, parts
