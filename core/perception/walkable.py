"""可通行区域评估：画面下半区域地面分割 → 分扇区评分 → 转向建议。

启发式（计划文档 3.2 节）：可通行 = 命中地面 HSV 颜色区间 且 非边缘（低纹理突变）。
阈值全部配置化，复杂材质（雪地/水面/暗场景）失效时实机校准（计划文档第 6 节）。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class WalkableParams:
    roi_top_ratio: float = 0.55  # 只分析该比例以下的画面（地面区域）
    ground_hsv_lower: tuple[int, int, int] = (0, 0, 40)
    ground_hsv_upper: tuple[int, int, int] = (179, 90, 220)
    edge_threshold: int = 80  # Canny 低阈值，高阈值取其 2 倍
    straight_min_score: float = 0.35  # 中央扇区达到该分即可直行


@dataclass(frozen=True)
class WalkableResult:
    """左/中/右三个扇区的可通行评分（0~1）与转向建议。"""

    left: float
    center: float
    right: float
    suggestion: str  # "left" | "straight" | "right"

    def as_dict(self) -> dict[str, float | str]:
        return {
            "left": self.left,
            "center": self.center,
            "right": self.right,
            "suggestion": self.suggestion,
        }


class WalkableAnalyzer:
    def __init__(self, params: WalkableParams | None = None):
        self.params = params or WalkableParams()

    def analyze(self, frame_bgr: np.ndarray) -> WalkableResult:
        p = self.params
        h = frame_bgr.shape[0]
        roi = frame_bgr[int(h * p.roi_top_ratio) :, :]
        if roi.size == 0:
            return WalkableResult(0.0, 0.0, 0.0, "straight")

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        ground = cv2.inRange(
            hsv,
            np.array(p.ground_hsv_lower, dtype=np.uint8),
            np.array(p.ground_hsv_upper, dtype=np.uint8),
        )
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, p.edge_threshold, p.edge_threshold * 2)
        free = (ground > 0) & (edges == 0)

        thirds = np.array_split(free, 3, axis=1)
        left, center, right = (float(sec.mean()) for sec in thirds)

        if center >= p.straight_min_score and center >= left and center >= right:
            suggestion = "straight"
        elif left >= right:
            suggestion = "left"
        else:
            suggestion = "right"
        return WalkableResult(left, center, right, suggestion)
