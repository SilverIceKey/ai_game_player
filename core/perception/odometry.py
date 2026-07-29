"""光流视觉里程计（LK 稀疏光流，CPU 可跑，不占显存）。

运动模型约定（保守路线，仅截屏）：

- 画面内容整体水平平移 ≈ 视角 yaw 转动（内容右移 → 视角左转，θ 减小；右转 θ 增大）
- 画面下半区域内容纵向流动 ≈ 前进/后退（内容下移 → 前进）
- 局部坐标系：启动点为原点，y 轴朝初始朝向，θ 为 yaw（弧度）

已知限制（计划文档第 6 节）：

- 平移与转身无法从纯平移光流中区分，strafe 被 yaw 吸收
- 弱纹理场景跟踪点不足时退化为静止
- 累积漂移不做回环校正（土地庙等特征点回环后置 M3）
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class OdometryParams:
    downscale: float = 0.5  # 降采样比例，控制 CPU 开销
    max_corners: int = 200
    quality_level: float = 0.01
    min_distance: int = 8
    yaw_per_pixel: float = 0.002  # 弧度 / 降采样后像素（实机校准）
    forward_per_pixel: float = 0.02  # 局部坐标单位 / 降采样后像素（实机校准）
    min_flow: float = 0.3  # 死区：中位流低于该值视为噪声
    min_tracked: int = 20  # 有效跟踪点下限，不足时本帧不更新位姿
    fb_error: float = 1.0  # 前向-后向验证容差（降采样后像素），过滤弱纹理虚假跟踪


@dataclass
class Pose:
    """航位推算位姿：局部坐标 (x, y) + 航向 θ（弧度）。"""

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.theta)


class VisualOdometry:
    """相邻帧 LK 光流 → 中位流统计 → 位姿积分。"""

    def __init__(self, params: OdometryParams | None = None):
        self.params = params or OdometryParams()
        self.pose = Pose()
        self._prev: np.ndarray | None = None

    def reset(self) -> None:
        self.pose = Pose()
        self._prev = None

    def update(self, frame_bgr: np.ndarray) -> Pose:
        p = self.params
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if p.downscale != 1.0:
            gray = cv2.resize(gray, None, fx=p.downscale, fy=p.downscale, interpolation=cv2.INTER_AREA)

        prev = self._prev
        self._prev = gray
        if prev is None:
            return self.pose

        pts = cv2.goodFeaturesToTrack(prev, p.max_corners, p.quality_level, p.min_distance)
        if pts is None or len(pts) < p.min_tracked:
            return self.pose
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(
            prev, gray, pts, None, winSize=(31, 31), maxLevel=3
        )
        if nxt is None:
            return self.pose
        # 前向-后向验证：弱纹理区域 LK 会"虚假跟踪"出随机流，往返误差大的点剔除
        back, status_back, _ = cv2.calcOpticalFlowPyrLK(
            gray, prev, nxt, None, winSize=(31, 31), maxLevel=3
        )
        if back is None:
            return self.pose
        ok = (status.reshape(-1) == 1) & (status_back.reshape(-1) == 1)
        if int(ok.sum()) < p.min_tracked:
            return self.pose
        fb = np.linalg.norm(
            pts[ok].reshape(-1, 2) - back[ok].reshape(-1, 2), axis=1
        )
        good = fb < p.fb_error
        pts_ok = pts[ok][good]
        nxt_ok = nxt[ok][good]
        if len(pts_ok) < p.min_tracked:
            return self.pose

        flow = (nxt_ok - pts_ok).reshape(-1, 2)
        src_y = pts_ok.reshape(-1, 2)[:, 1]
        dx = float(np.median(flow[:, 0]))
        bottom = src_y >= gray.shape[0] * 0.5
        dy_bottom = float(np.median(flow[bottom, 1])) if bool(bottom.any()) else 0.0

        if abs(dx) >= p.min_flow:
            self.pose.theta += -dx * p.yaw_per_pixel
        if abs(dy_bottom) >= p.min_flow:
            forward = dy_bottom * p.forward_per_pixel
            self.pose.x += forward * math.sin(self.pose.theta)
            self.pose.y += forward * math.cos(self.pose.theta)
        return self.pose
