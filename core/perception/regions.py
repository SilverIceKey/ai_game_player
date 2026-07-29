"""HUD 区域颜色/阈值检测工具（轻量，CPU 可跑）。

所有区域坐标基于基准分辨率（默认 1920x1080）定义，运行时按实际帧大小等比缩放；
区域坐标与 HSV 阈值全部来自配置，首次实机需校准（计划文档第 5 节假设 3）。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ColorRange:
    """HSV 颜色范围（OpenCV H∈[0,179]）。"""

    lower: tuple[int, int, int]
    upper: tuple[int, int, int]


@dataclass(frozen=True)
class BarSpec:
    """条状 HUD（血条/体力条）：rect=(x, y, w, h)，按列统计填充比例。"""

    rect: tuple[int, int, int, int]
    color: ColorRange
    column_fill: float = 0.3  # 单列匹配像素占比达到该值视为该列已填充


@dataclass(frozen=True)
class PresenceSpec:
    """块状 HUD（葫芦/死亡提示）：匹配色像素占比超阈值即视为存在。"""

    rect: tuple[int, int, int, int]
    color: ColorRange
    min_ratio: float = 0.05


def scale_rect(
    rect: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
    base_resolution: tuple[int, int] = (1920, 1080),
) -> tuple[int, int, int, int]:
    """把基准分辨率下的 rect 等比缩放到实际帧大小，并裁剪到帧内。"""
    h, w = frame_shape[:2]
    bw, bh = base_resolution
    sx, sy = w / bw, h / bh
    x, y, rw, rh = rect
    x0 = min(w - 1, max(0, int(round(x * sx))))
    y0 = min(h - 1, max(0, int(round(y * sy))))
    x1 = min(w, max(x0 + 1, int(round((x + rw) * sx))))
    y1 = min(h, max(y0 + 1, int(round((y + rh) * sy))))
    return x0, y0, x1 - x0, y1 - y0


def crop_region(
    frame: np.ndarray,
    rect: tuple[int, int, int, int],
    base_resolution: tuple[int, int] = (1920, 1080),
) -> np.ndarray:
    x, y, w, h = scale_rect(rect, frame.shape, base_resolution)
    return frame[y : y + h, x : x + w]


def _mask(region: np.ndarray, color: ColorRange) -> np.ndarray:
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    return cv2.inRange(
        hsv,
        np.array(color.lower, dtype=np.uint8),
        np.array(color.upper, dtype=np.uint8),
    )


def measure_bar(
    frame: np.ndarray,
    spec: BarSpec,
    base_resolution: tuple[int, int] = (1920, 1080),
) -> float:
    """测量条状 HUD 的填充比例，返回 [0, 1]。"""
    region = crop_region(frame, spec.rect, base_resolution)
    if region.size == 0:
        return 0.0
    mask = _mask(region, spec.color)
    column_fill = (mask > 0).mean(axis=0)
    return float((column_fill >= spec.column_fill).mean())


def detect_presence(
    frame: np.ndarray,
    spec: PresenceSpec,
    base_resolution: tuple[int, int] = (1920, 1080),
) -> bool:
    """检测块状 HUD 是否存在（匹配色像素占比 >= spec.min_ratio）。"""
    region = crop_region(frame, spec.rect, base_resolution)
    if region.size == 0:
        return False
    mask = _mask(region, spec.color)
    return float((mask > 0).mean()) >= spec.min_ratio
