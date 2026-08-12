"""runtime/preprocess.py 单元测试（spec §14 模型分辨率 / §30 Preprocess Worker）。"""
from __future__ import annotations

import numpy as np
import pytest

from runtime.preprocess import preprocess_frame


class TestPreprocessFrame:
    def test_resize_and_normalize(self):
        frame = np.full((1080, 1920, 3), 255, dtype=np.uint8)
        out = preprocess_frame(frame, 384, 216)
        assert out.shape == (216, 384, 3)
        assert out.dtype == np.float32
        assert out.min() == pytest.approx(1.0)
        assert out.max() == pytest.approx(1.0)

    def test_value_range_zero_to_one(self):
        frame = np.random.default_rng(0).integers(0, 256, (100, 200, 3), dtype=np.uint8)
        out = preprocess_frame(frame, 64, 32)
        assert 0.0 <= out.min() <= out.max() <= 1.0

    def test_mid_gray_maps_to_half(self):
        frame = np.full((10, 10, 3), 128, dtype=np.uint8)
        out = preprocess_frame(frame, 10, 10)
        assert out[0, 0, 0] == pytest.approx(128 / 255.0)

    def test_rejects_non_bgr_shape(self):
        with pytest.raises(ValueError):
            preprocess_frame(np.zeros((100, 100), dtype=np.uint8), 64, 64)

    def test_rejects_invalid_size(self):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            preprocess_frame(frame, 0, 216)
