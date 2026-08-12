"""Temporal Encoder / Policy Decoder 接口定义（spec §16）。

架构位置（spec §16）：

    Visual Tokens ──→ Temporal Encoder ──→ Policy Decoder ──→ Action Chunk
    Action History ──┘

- Temporal Encoder：融合 visual tokens 与 Action History（spec §8.2），
  建模时序上下文；
- Policy Decoder：输出未来 N 步 Action Chunk（spec §15，默认 4 步 × 50ms）。

规模建议（spec §17）：Temporal / Policy 合计 100M~300M 参数。

本轮不引入 PyTorch，只定义协议；torch 实现落地时按上述约束编写。
"""
from __future__ import annotations

from typing import Protocol

from capture.action import NormalizedAction


class TemporalEncoder(Protocol):
    """融合视觉 token 序列与动作历史，产出时序上下文表示。"""

    def encode(
        self,
        visual_tokens: object,
        action_history: list[NormalizedAction],
    ) -> object:
        """返回时序上下文表示（返回类型由 torch 实现定义为 Tensor）。"""
        ...


class PolicyDecoder(Protocol):
    """由时序上下文解码未来 Action Chunk（spec §15）。

    输出必须经由 model/action_heads.py 的拆分 Head 产生（spec §19），
    禁止单一巨大 action class 直接输出。
    """

    def decode(self, context: object, future_action_steps: int) -> object:
        """解码 future_action_steps 步动作（返回类型由 torch 实现定义）。"""
        ...
