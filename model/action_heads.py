"""Action Head 接口定义（spec §19：拆 Head，禁止单一巨大 action class）。

三类 Head（输出共同拼成 NormalizedAction）：

- Movement Head（§19.1）：move_x / move_y 连续回归（或 discretized bins），
  范围 [-1, 1]；
- Camera Head（§19.2）：camera_x / camera_y。优先 Discretized Distribution，
  不建议纯 MSE 回归——左右转向目标（-0.8 / +0.8）在 MSE 下会互抵消为 0；
- Button Head（§19.3）：multi-label binary heads，每个按钮一个二分类输出。
  不能把所有组合动作编码成一个巨大 action class。

本轮不引入 PyTorch，只定义协议；torch 实现落地时按上述约束编写。
"""
from __future__ import annotations

from typing import Protocol


class MovementHead(Protocol):
    """spec §19.1：输出 move_x / move_y（[-1, 1]，回归或 discretized bins）。"""

    def forward(self, features: object) -> object:
        """输出两个连续轴（回归）或 bins 分布；具体类型由 torch 实现定义。"""
        ...


class CameraHead(Protocol):
    """spec §19.2：输出 camera_x / camera_y。

    优先实验 Discretized Distribution，避免双向目标在 MSE 下互抵消。
    """

    def forward(self, features: object) -> object:
        """输出相机轴分布或连续值；具体类型由 torch 实现定义。"""
        ...


class ButtonHead(Protocol):
    """spec §19.3：multi-label binary heads。

    每个按钮独立二分类（多标签，可同时按下），
    输出维度 = len(capture.action.BUTTONS)。
    """

    def forward(self, features: object) -> object:
        """输出各按钮 logit/概率序列；具体类型由 torch 实现定义。"""
        ...
