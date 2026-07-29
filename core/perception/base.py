"""感知层接口契约：屏幕帧来源。

合规约束（计划文档第 2 节）：仅截屏，不读内存、不注入。
硬件约束：实时感知模型必须与游戏共存于 RTX 2070s（8GB 显存），
只许轻量模型（小型检测/分类网络、OCR、模板匹配）。
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class FrameSource(Protocol):
    """屏幕帧来源。实现方负责窗口定位与截屏。"""

    def grab(self) -> np.ndarray:
        """抓取一帧 BGR 图像，形状 (H, W, 3)。"""
        ...
