"""动态血条检测：普通小怪血条浮在怪物头顶、位置不固定（计划 3.1a 节）。

在可配置搜索区域内做 HSV 颜色阈值 → 轮廓检测 → 形状筛选
（细长水平条：最小长度 / 长宽比 / 填充均匀度，全部走配置）→
取离画面中心最近的一条作为当前目标血条。Boss 固定血条仍走 regions.py 的
固定 ROI（games/wukong/adapter.py 中 Boss 条优先）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from core.perception.regions import ColorRange, color_mask, scale_rect


@dataclass(frozen=True)
class BarSearchSpec:
    """动态血条搜索参数：搜索区域 + HSV 阈值 + 形状筛选（基准分辨率下定义）。

    小怪血条 = 填充色（血量）+ 背景槽（全长）。track_color 给定背景槽颜色范围时，
    轮廓检测作用在背景槽上（定全长），ratio = 填充色在槽内的按列填充比例；
    未配置 track_color 时退化为单色轮廓检测（ratio 语义为轮廓内填充均匀度）。
    """

    rect: tuple[int, int, int, int]  # 搜索区域 (x, y, w, h)
    color: ColorRange  # 填充色（血量部分）
    track_color: ColorRange | None = None  # 背景槽（全长部分），可选
    min_length: int = 40  # 条最小长度（像素，基准分辨率）
    min_aspect: float = 3.0  # 长宽比下限（细长水平条）
    min_fill: float = 0.5  # 轮廓包围盒内匹配像素占比下限（填充均匀度）
    column_fill: float = 0.3  # 条填充比例按列统计阈值（同 BarSpec 语义）


@dataclass(frozen=True)
class DetectedBar:
    """检出的一条动态血条。"""

    ratio: float  # 填充比例 [0, 1]
    box: tuple[int, int, int, int]  # 帧坐标 (x, y, w, h)
    center_distance: float  # 条中心到画面中心的归一化距离（0~1，越小越靠近准星）


def detect_bar(
    frame: np.ndarray,
    spec: BarSearchSpec,
    base_resolution: tuple[int, int] = (1920, 1080),
) -> DetectedBar | None:
    """在搜索区域内检测最靠近画面中心的血条；未检出返回 None。"""
    x0, y0, rw, rh = scale_rect(spec.rect, frame.shape, base_resolution)
    region = frame[y0 : y0 + rh, x0 : x0 + rw]
    if region.size == 0:
        return None
    fill_mask = color_mask(region, spec.color)
    # 轮廓检测作用在「槽 ∪ 填充」并集上（槽定全长，填充段不会把轮廓截断）；
    # 未配置背景槽颜色时退化为单色填充轮廓
    if spec.track_color is not None:
        contour_mask = color_mask(region, spec.track_color) | fill_mask
    else:
        contour_mask = fill_mask

    contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    sx = frame.shape[1] / base_resolution[0]
    min_length = spec.min_length * sx
    fh, fw = frame.shape[:2]
    cx, cy = fw / 2.0, fh / 2.0
    norm = math.hypot(cx, cy) or 1.0

    best: DetectedBar | None = None
    for contour in contours:
        bx, by, bw, bh = cv2.boundingRect(contour)
        if bw < min_length or bw / max(bh, 1) < spec.min_aspect:
            continue
        sub_contour = contour_mask[by : by + bh, bx : bx + bw]
        if sub_contour.size == 0 or float((sub_contour > 0).mean()) < spec.min_fill:
            continue
        sub_fill = fill_mask[by : by + bh, bx : bx + bw]
        column_fill = (sub_fill > 0).mean(axis=0)
        ratio = float((column_fill >= spec.column_fill).mean())
        gx, gy = x0 + bx + bw / 2.0, y0 + by + bh / 2.0
        distance = math.hypot(gx - cx, gy - cy) / norm
        if best is None or distance < best.center_distance:
            best = DetectedBar(ratio=ratio, box=(x0 + bx, y0 + by, bw, bh), center_distance=distance)
    return best
