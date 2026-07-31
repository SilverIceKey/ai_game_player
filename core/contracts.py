"""跨层共享数据契约。

分层约定（docs/plans/PLAN-20260729-project-skeleton-v1.md 第 3 节）：
apps → games → core，禁止反向依赖。共享契约因此放在 core，games 可引用。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GameState:
    """一帧画面解析出的标准游戏状态。

    各游戏的具体字段（血量、体力、敌人位置等）由适配器定义，放入 raw。
    M3 追加（仅追加不破坏）：frame_id 帧序号；confidence 感知置信度透传
    （hp_visible 等信号换算，有血条读数=高置信，隐藏降级）。
    """

    timestamp: float
    scene: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
    frame_id: int = 0
    confidence: dict[str, float] = field(default_factory=dict)


@dataclass
class Action:
    """可执行动作契约。control 层只认识此结构，不认识具体游戏。"""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Suggestion:
    """半自动模式下给玩家的决策建议（人确认后执行）。"""

    action: Action
    reason: str
    confidence: float = 0.0
