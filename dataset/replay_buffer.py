"""Replay Buffer：四类桶加权采样（spec §28 + §25）。

四个类别（与 config.SamplingConfig 的权重字段一一对应）：
- historical：历史 session 积累的普通样本
- recent：最近 session 的新样本
- correction：§26/§27 人工接管后的纠正样本（DAgger 闭环核心）
- rare：稀有事件样本，含 §25 Recovery Data 语义（卡墙/死亡/闪避失败等，
  由调用方在 add 时标记，本模块不做自动判定）

采样策略：
- sample(n, weights) 有放回抽样：先按类别权重选桶，再桶内均匀随机。
- 空桶权重剔除后按剩余桶归一化（避免抽到空桶导致崩溃或偏斜）。
- random.Random 可注入，测试用固定种子保证可复现。
"""
from __future__ import annotations

import random
from typing import Any

from config import SamplingConfig

CATEGORY_HISTORICAL = "historical"
CATEGORY_RECENT = "recent"
CATEGORY_CORRECTION = "correction"
CATEGORY_RARE = "rare"
CATEGORIES: tuple[str, ...] = (
    CATEGORY_HISTORICAL,
    CATEGORY_RECENT,
    CATEGORY_CORRECTION,
    CATEGORY_RARE,
)


class ReplayBuffer:
    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng if rng is not None else random.Random()
        self._buckets: dict[str, list[Any]] = {name: [] for name in CATEGORIES}

    def add(self, sample: Any, category: str) -> None:
        if category not in self._buckets:
            raise ValueError(f"未知样本类别: {category!r}（合法值: {CATEGORIES}）")
        self._buckets[category].append(sample)

    def size(self) -> int:
        return sum(len(bucket) for bucket in self._buckets.values())

    def category_size(self, category: str) -> int:
        if category not in self._buckets:
            raise ValueError(f"未知样本类别: {category!r}（合法值: {CATEGORIES}）")
        return len(self._buckets[category])

    def sample(self, n: int, weights: SamplingConfig) -> list[Any]:
        """按权重有放回抽 n 条。空桶权重归一化；整个 buffer 为空返回 []。"""
        if n < 0:
            raise ValueError(f"n 必须为非负整数: {n}")
        if n == 0 or self.size() == 0:
            return []

        raw = {
            CATEGORY_HISTORICAL: weights.historical,
            CATEGORY_RECENT: weights.recent,
            CATEGORY_CORRECTION: weights.correction,
            CATEGORY_RARE: weights.rare,
        }
        for name, w in raw.items():
            if w < 0:
                raise ValueError(f"采样权重不能为负: {name}={w}")
        # 剔除空桶，剩余桶权重归一化
        active = {name: w for name, w in raw.items() if w > 0 and self._buckets[name]}
        if not active:
            raise ValueError(
                f"所有非空桶的采样权重均为 0，无法抽样: {dict(raw)}"
            )
        total = sum(active.values())
        names = list(active)
        probs = [w / total for w in active.values()]

        out: list[Any] = []
        for _ in range(n):
            bucket_name = self._rng.choices(names, weights=probs, k=1)[0]
            out.append(self._rng.choice(self._buckets[bucket_name]))
        return out
