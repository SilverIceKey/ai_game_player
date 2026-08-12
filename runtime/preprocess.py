"""帧预处理（spec §30 Preprocess Worker；§14 模型分辨率；§31 目标 <5ms）。

模型输入统一为 BGR float32 / 255（不做通道交换与均值方差归一化，
归一化策略属于模型侧，后续由 model/ 包在 encoder 内决定）。
"""
from __future__ import annotations

import cv2
import numpy as np


def preprocess_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """BGR uint8 帧 → resize 到 (width, height) → float32 / 255，值域 [0, 1]。"""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"输入帧必须是 (H, W, 3) 的 BGR 图像，实际 shape: {frame.shape}")
    if width <= 0 or height <= 0:
        raise ValueError(f"目标尺寸必须为正整数: width={width}, height={height}")
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0
