"""动作/帧的张量编解码（训练与推理共享，避免 train/inference 预处理漂移）。

本模块只用 numpy，不依赖 torch：张量转换发生在调用侧边界
（train/dataset.py 与 model/torch_policy.py）。

- NormalizedAction ↔ 定长向量（4 连续轴 + 14 按钮 0/1，顺序固定禁止改动）
- Camera 轴 ↔ 离散 bin（spec §19.2：离散分布优先，防左右互抵消）
- 帧归一化：HWC float[0,1] → CHW ImageNet mean/std（backbone 为预训练 ResNet）
"""
from __future__ import annotations

import numpy as np

from capture.action import BUTTONS, NormalizedAction

ACTION_DIM = 4 + len(BUTTONS)  # 18：move_x/y + camera_x/y + 按钮

# ImageNet 归一化参数（预训练 backbone 的输入约定）
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def action_to_vector(action: NormalizedAction) -> np.ndarray:
    """NormalizedAction → (18,) float32：4 轴原值 + 按钮 0/1（按 BUTTONS 顺序）。"""
    vec = np.zeros(ACTION_DIM, dtype=np.float32)
    vec[0] = action.move_x
    vec[1] = action.move_y
    vec[2] = action.camera_x
    vec[3] = action.camera_y
    for i, name in enumerate(BUTTONS):
        vec[4 + i] = 1.0 if name in action.buttons else 0.0
    return vec


def vector_to_action(vec: np.ndarray) -> NormalizedAction:
    """连续向量 → NormalizedAction（按钮阈值 0.5；camera 轴为连续值）。

    仅用于不需要 camera bin 分布语义的场景（调试/导出）。
    """
    vec = np.asarray(vec, dtype=np.float32)
    if vec.shape != (ACTION_DIM,):
        raise ValueError(f"动作向量维度必须为 ({ACTION_DIM},)，实际 {vec.shape}")
    buttons = frozenset(name for i, name in enumerate(BUTTONS) if vec[4 + i] > 0.5)
    return NormalizedAction(
        move_x=float(vec[0]), move_y=float(vec[1]),
        camera_x=float(vec[2]), camera_y=float(vec[3]),
        buttons=buttons,
    )


def camera_to_bin(value: float, bins: int) -> int:
    """[-1, 1] 连续值 → bin 下标 [0, bins)。bins 为奇数（含 0 位），构造期已校验。"""
    clipped = max(-1.0, min(1.0, float(value)))
    return int(round((clipped + 1.0) / 2.0 * (bins - 1)))


def bin_probs_to_camera(probs: np.ndarray) -> float:
    """bin 概率分布 → 连续值（期望，比 argmax 平滑）。"""
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 1 or probs.shape[0] < 3 or probs.shape[0] % 2 == 0:
        raise ValueError(f"bin 概率必须为奇数长度一维数组，实际 {probs.shape}")
    centers = np.linspace(-1.0, 1.0, probs.shape[0])
    total = probs.sum()
    if total <= 0:
        return 0.0
    return float((centers * probs).sum() / total)


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """HWC float32 [0,1]（runtime/preprocess 输出）→ CHW ImageNet 归一化 float32。"""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"帧形状必须为 (H, W, 3)，实际 {frame.shape}")
    chw = np.ascontiguousarray(frame.transpose(2, 0, 1), dtype=np.float32)
    # OpenCV 帧为 BGR，ImageNet 预训练约定 RGB：通道翻转
    chw = chw[::-1]
    return (chw - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
