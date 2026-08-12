"""dataset/replay_buffer.py 单元测试：权重分布（固定种子）与空桶归一化。"""
from __future__ import annotations

import random
from collections import Counter

import pytest

from config import SamplingConfig
from dataset.replay_buffer import CATEGORIES, ReplayBuffer


def _make_buffer(counts: dict[str, int], seed: int = 42) -> ReplayBuffer:
    buf = ReplayBuffer(rng=random.Random(seed))
    for category, n in counts.items():
        for i in range(n):
            buf.add(f"{category}-{i}", category)
    return buf


def test_add_and_sizes():
    buf = _make_buffer({"historical": 3, "rare": 2})
    assert buf.size() == 5
    assert buf.category_size("historical") == 3
    assert buf.category_size("rare") == 2
    assert buf.category_size("recent") == 0


def test_invalid_category_raises():
    buf = ReplayBuffer()
    with pytest.raises(ValueError, match="未知样本类别"):
        buf.add("x", "unknown")
    with pytest.raises(ValueError, match="未知样本类别"):
        buf.category_size("unknown")


def test_weight_distribution_seeded():
    buf = _make_buffer({"historical": 10, "recent": 10, "correction": 10, "rare": 10})
    weights = SamplingConfig(historical=0.50, recent=0.25, correction=0.20, rare=0.05)
    samples = buf.sample(10_000, weights)
    assert len(samples) == 10_000

    counter = Counter(s.rsplit("-", 1)[0] for s in samples)
    total = sum(counter.values())
    ratios = {name: counter[name] / total for name in CATEGORIES}
    # 统计容差 ±0.03（种子固定，分布应紧贴权重）
    assert abs(ratios["historical"] - 0.50) < 0.03
    assert abs(ratios["recent"] - 0.25) < 0.03
    assert abs(ratios["correction"] - 0.20) < 0.03
    assert abs(ratios["rare"] - 0.05) < 0.03


def test_empty_bucket_weight_renormalized():
    # rare 桶为空：权重 0.05 剔除，剩余 0.50/0.25/0.20 归一化
    buf = _make_buffer({"historical": 10, "recent": 10, "correction": 10})
    weights = SamplingConfig(historical=0.50, recent=0.25, correction=0.20, rare=0.05)
    samples = buf.sample(10_000, weights)

    counter = Counter(s.rsplit("-", 1)[0] for s in samples)
    assert "rare" not in counter
    total = sum(counter.values())
    scale = 0.50 + 0.25 + 0.20
    assert abs(counter["historical"] / total - 0.50 / scale) < 0.03
    assert abs(counter["recent"] / total - 0.25 / scale) < 0.03
    assert abs(counter["correction"] / total - 0.20 / scale) < 0.03


def test_seeded_reproducible():
    kwargs = {"historical": 5, "recent": 5, "correction": 5, "rare": 5}
    buf1 = _make_buffer(kwargs, seed=7)
    buf2 = _make_buffer(kwargs, seed=7)
    weights = SamplingConfig()
    assert buf1.sample(50, weights) == buf2.sample(50, weights)


def test_sample_with_replacement():
    buf = _make_buffer({"rare": 1}, seed=1)
    weights = SamplingConfig(historical=0, recent=0, correction=0, rare=1.0)
    samples = buf.sample(5, weights)
    assert samples == ["rare-0"] * 5  # 桶里只有一条，有放回抽样全中


def test_empty_buffer_returns_empty():
    buf = ReplayBuffer(rng=random.Random(0))
    assert buf.sample(10, SamplingConfig()) == []
    assert buf.sample(0, SamplingConfig()) == []


def test_all_zero_weights_on_nonempty_raises():
    buf = _make_buffer({"historical": 2})
    with pytest.raises(ValueError, match="权重均为 0"):
        buf.sample(1, SamplingConfig(historical=0, recent=0, correction=0, rare=0))


def test_negative_weight_raises():
    buf = _make_buffer({"historical": 2})
    with pytest.raises(ValueError, match="不能为负"):
        buf.sample(1, SamplingConfig(historical=-0.1, recent=0.5, correction=0.5, rare=0))
