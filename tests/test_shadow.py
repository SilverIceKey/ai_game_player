"""evaluation/shadow.py 单元测试：SHADOW 对齐指标累计与汇总（spec §41）。"""
from __future__ import annotations

import pytest

from capture.action import NormalizedAction
from evaluation.shadow import ShadowMetrics


def test_empty_summary() -> None:
    m = ShadowMetrics()
    s = m.summary()
    assert s["sample_count"] == 0
    assert s["move_mse"] == 0.0
    assert s["camera_mse"] == 0.0


def test_summary_hand_computed() -> None:
    m = ShadowMetrics()
    # 第 1 对：AI 想前走+闪避，玩家前走+闪避 → 完全对齐，move 轴误差 0
    m.update(
        NormalizedAction(move_y=1.0, buttons={"dodge"}),
        NormalizedAction(move_y=1.0, buttons={"dodge"}),
    )
    # 第 2 对：AI 想右移 0.5 + 喝药，玩家原地 + 无按钮 → move 误差 (0.25+0)/2
    m.update(
        NormalizedAction(move_x=0.5, camera_x=0.4, buttons={"heal"}),
        NormalizedAction.neutral(),
    )
    s = m.summary()
    assert s["sample_count"] == 2
    # move_mse = (0 + (0.25+0)/2) / 2 = 0.0625
    assert s["move_mse"] == pytest.approx(0.0625)
    # camera_mse = (0 + (0.16+0)/2) / 2 = 0.04
    assert s["camera_mse"] == pytest.approx(0.04)
    # dodge: tp=1 → P=R=1；heal: fp=1 → P=0
    assert s["buttons"]["dodge"]["precision"] == 1.0
    assert s["buttons"]["dodge"]["recall"] == 1.0
    assert s["buttons"]["heal"]["precision"] == 0.0
    assert s["buttons"]["heal"]["fp"] == 1


def test_render_contains_metrics() -> None:
    m = ShadowMetrics()
    m.update(NormalizedAction(buttons={"dodge"}), NormalizedAction.neutral())
    text = m.render()
    assert "shadow metrics (samples=1)" in text
    assert "dodge" in text
    assert "precision" in text
