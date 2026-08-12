"""Visual Encoder 接口定义（spec §16/§17/§18）。

架构位置（spec §16）：

    Video Frames → Visual Encoder → Visual Tokens → Temporal Encoder → ...

规模建议（spec §17）：Visual Encoder 50M~150M 参数；效果差时按
Timestamp → Labels → Dataset → ... → Model Capacity 顺序排查，
禁止第一反应堆大模型。

训练策略（spec §18，三阶段，本文件以协议形式固化阶段契约）：
- Stage 1: Freeze Visual Backbone
- Stage 2: Unfreeze Last Blocks
- Stage 3: Optional Full Fine-tuning

禁止从随机初始化训练完整视觉系统，必须使用 Pretrained Visual Encoder。

本轮不引入 PyTorch，只定义协议；torch 实现落地时按上述约束编写。
"""
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np


class VisualTrainStage(Enum):
    """spec §18 三阶段训练策略。"""

    FREEZE_BACKBONE = "freeze_backbone"  # Stage 1：冻结视觉主干
    UNFREEZE_LAST_BLOCKS = "unfreeze_last_blocks"  # Stage 2：解冻最后若干 block
    FULL_FINETUNE = "full_finetune"  # Stage 3：可选全量微调


class VisualEncoder(Protocol):
    """Video History 帧序列 → Visual Tokens。

    实现约束：
    - 必须基于预训练权重初始化（spec §18：禁止随机初始化训视觉）；
    - 训练时按 VisualTrainStage 控制冻结范围；
    - 输入为预处理后的帧（spec §14：384×216），输出 token 序列供
      Temporal Encoder 消费。
    """

    def encode_frames(self, frames: list[np.ndarray]) -> object:
        """编码帧窗口为 visual tokens（返回类型由 torch 实现定义为 Tensor）。"""
        ...

    def apply_train_stage(self, stage: VisualTrainStage) -> None:
        """按 spec §18 设置当前训练阶段的冻结策略。"""
        ...
