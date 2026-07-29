"""光流里程计测试：合成平移（yaw）与缩放（前进）帧。"""
import cv2
import numpy as np
import pytest

from core.perception.odometry import OdometryParams, VisualOdometry

SIZE = (720, 1280)  # H, W


def _texture(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gray = rng.integers(0, 256, SIZE, dtype=np.uint8)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # 形成可跟踪的斑点纹理
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _make_odometry(**overrides) -> VisualOdometry:
    params = OdometryParams(downscale=0.5, min_flow=0.1, **overrides)
    return VisualOdometry(params)


def test_first_frame_no_motion():
    odo = _make_odometry()
    pose = odo.update(_texture(1))
    assert (pose.x, pose.y, pose.theta) == (0.0, 0.0, 0.0)


def test_yaw_from_horizontal_pan():
    """画面内容右移 → 视角左转（θ 减小），幅度符合 yaw_per_pixel。"""
    odo = _make_odometry(yaw_per_pixel=0.002)
    f1 = _texture(1)
    shift_px = 16
    M = np.float32([[1, 0, shift_px], [0, 1, 0]])
    f2 = cv2.warpAffine(f1, M, (SIZE[1], SIZE[0]))
    odo.update(f1)
    pose = odo.update(f2)
    expected = -shift_px * 0.5 * 0.002  # 降采样 0.5
    assert pose.theta == pytest.approx(expected, rel=0.1)
    # 纯水平平移不应产生前进
    assert pose.y == pytest.approx(0.0, abs=1e-9)


def test_forward_from_zoom_in():
    """画面内容放大（下半区域下移）→ 判定为前进（y 增大）。"""
    odo = _make_odometry()
    f1 = _texture(2)
    center = (SIZE[1] / 2.0, SIZE[0] / 2.0)
    M = cv2.getRotationMatrix2D(center, 0.0, 1.06)
    f2 = cv2.warpAffine(f1, M, (SIZE[1], SIZE[0]))
    odo.update(f1)
    pose = odo.update(f2)
    assert pose.y > 0.0
    # 估计器有残差：缩放产生的水平流中位数不完全为 0，x 只要求接近 0
    assert pose.x == pytest.approx(0.0, abs=1e-3)


def test_backward_from_zoom_out():
    # 缩小时部分角点跟丢，放宽有效跟踪点下限
    odo = _make_odometry(min_tracked=5)
    f1 = _texture(3)
    center = (SIZE[1] / 2.0, SIZE[0] / 2.0)
    M = cv2.getRotationMatrix2D(center, 0.0, 0.94)
    f2 = cv2.warpAffine(f1, M, (SIZE[1], SIZE[0]))
    odo.update(f1)
    pose = odo.update(f2)
    assert pose.y < 0.0


def test_textureless_frame_degrades_to_static():
    """弱纹理（纯色）帧跟踪点不足 → 位姿不更新，不发散。"""
    odo = _make_odometry()
    flat = np.full((*SIZE, 3), 128, dtype=np.uint8)
    odo.update(_texture(4))
    pose = odo.update(flat)
    assert (pose.x, pose.y, pose.theta) == (0.0, 0.0, 0.0)


def test_reset():
    odo = _make_odometry()
    f1 = _texture(5)
    odo.update(f1)
    odo.update(cv2.warpAffine(f1, np.float32([[1, 0, 10], [0, 1, 0]]), (SIZE[1], SIZE[0])))
    odo.reset()
    assert (odo.pose.x, odo.pose.y, odo.pose.theta) == (0.0, 0.0, 0.0)
