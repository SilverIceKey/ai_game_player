"""model/encoding.py 单元测试：动作向量、camera bin、帧归一化。"""
from __future__ import annotations

import numpy as np
import pytest

from capture.action import BUTTONS, NormalizedAction
from model.encoding import (
    ACTION_DIM,
    action_to_vector,
    bin_probs_to_camera,
    camera_to_bin,
    normalize_frame,
    vector_to_action,
)


def test_action_vector_layout() -> None:
    action = NormalizedAction(
        move_x=0.5, move_y=-1.0, camera_x=0.25, camera_y=-0.25,
        buttons=frozenset({"dodge", "heal"}),
    )
    vec = action_to_vector(action)
    assert vec.shape == (ACTION_DIM,)
    assert vec[0] == 0.5 and vec[1] == -1.0
    assert vec[2] == 0.25 and vec[3] == -0.25
    assert vec[4 + BUTTONS.index("dodge")] == 1.0
    assert vec[4 + BUTTONS.index("heal")] == 1.0
    assert vec[4:].sum() == 2.0


def test_vector_roundtrip() -> None:
    action = NormalizedAction(move_x=0.3, buttons=frozenset({"jump"}))
    back = vector_to_action(action_to_vector(action))
    assert back.move_x == pytest.approx(0.3)
    assert back.buttons == frozenset({"jump"})


def test_camera_bin_roundtrip() -> None:
    bins = 21
    for value in (-1.0, -0.5, 0.0, 0.5, 1.0):
        idx = camera_to_bin(value, bins)
        assert 0 <= idx < bins
        assert idx == camera_to_bin(value, bins)  # 幂等
    assert camera_to_bin(0.0, bins) == bins // 2  # 0 在中位
    assert camera_to_bin(-1.0, bins) == 0
    assert camera_to_bin(1.0, bins) == bins - 1
    # 越界截断
    assert camera_to_bin(5.0, bins) == bins - 1


def test_bin_probs_expectation() -> None:
    probs = np.zeros(21)
    probs[10] = 1.0
    assert bin_probs_to_camera(probs) == pytest.approx(0.0)
    probs = np.zeros(21)
    probs[20] = 1.0
    assert bin_probs_to_camera(probs) == pytest.approx(1.0)
    # 对称分布 → 期望居中
    probs = np.zeros(21)
    probs[0] = probs[20] = 0.5
    assert bin_probs_to_camera(probs) == pytest.approx(0.0)


def test_normalize_frame_shape_and_bgr_flip() -> None:
    frame = np.zeros((36, 64, 3), dtype=np.float32)
    frame[:, :, 0] = 1.0  # BGR 的 B 通道
    out = normalize_frame(frame)
    assert out.shape == (3, 36, 64)
    # BGR→RGB 翻转后，原 B 通道（值 1.0）落在 RGB 的 B 位（下标 2）：(1-0.406)/0.225 ≈ 2.64
    assert out[2].max() > 1.0
    assert out[0].max() < 0.0  # 原 R 通道为 0：(0-0.485)/0.229 ≈ -2.12


def test_normalize_frame_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="帧形状"):
        normalize_frame(np.zeros((36, 64), dtype=np.float32))
