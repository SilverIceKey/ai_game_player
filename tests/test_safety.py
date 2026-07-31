"""M3 安全层测试：F12 急停 toggle、失焦保护、非 Windows 默认、急停阻断主循环输入。"""
import json
from pathlib import Path

import numpy as np
import pytest

from apps.auto_player import main as auto_main
from core.config import (
    GameRef,
    LLMConfig,
    RecorderConfig,
    RuntimeConfig,
    Settings,
)
from core.contracts import Action, GameState
from core.safety import SafetyMonitor, SafetyParams, SafetyState, vk_for_key
from games.wukong.adapter import WukongConfig


def _monitor(presses=(), focused=True, stop_on_focus_lost=True, releases=None):
    queue = iter(presses)
    return SafetyMonitor(
        SafetyParams(stop_on_focus_lost=stop_on_focus_lost),
        on_release=(lambda: releases.append(1)) if releases is not None else None,
        key_poller=lambda key: next(queue, False),
        focus_checker=lambda: focused,
    )


def test_vk_mapping():
    assert vk_for_key("F12") == 0x7B
    assert vk_for_key("F1") == 0x70
    assert vk_for_key("a") == ord("A")
    with pytest.raises(Exception, match="emergency_stop_key"):
        vk_for_key("NOT_A_KEY")


def test_invalid_key_rejected_at_init():
    from core.config import ConfigError
    with pytest.raises(ConfigError, match="emergency_stop_key"):
        SafetyMonitor(SafetyParams(emergency_stop_key="NOT_A_KEY"))


def test_emergency_toggle_and_release():
    releases: list = []
    mon = _monitor(presses=[True, True, False, True], releases=releases)
    s1 = mon.check()
    assert s1.emergency_stopped and "F12" in s1.detail  # 按下沿 → 急停
    s2 = mon.check()
    assert s2.emergency_stopped  # 按住不重复翻转
    s3 = mon.check()
    assert s3.emergency_stopped  # 松开后保持急停
    s4 = mon.check()
    assert not s4.emergency_stopped  # 再按一次 → 恢复
    assert len(releases) == 3  # 急停期间每次 check 都释放输入


def test_focus_lost_releases_and_blocks():
    releases: list = []
    mon = _monitor(focused=False, releases=releases)
    state = mon.check()
    assert state.focus_lost and not state.emergency_stopped
    assert "失焦" in state.detail
    assert releases


def test_focus_check_disabled_by_config():
    mon = _monitor(focused=False, stop_on_focus_lost=False)
    assert mon.check() == SafetyState(False, False, "")


def test_defaults_safe_on_non_windows():
    import sys
    if sys.platform == "win32":
        pytest.skip("本用例仅覆盖非 Windows 默认 poller/checker")
    mon = SafetyMonitor(SafetyParams(), window_title="任意窗口")
    state = mon.check()
    assert not state.emergency_stopped and not state.focus_lost


# ---------- 主循环集成：急停状态下拒绝一切动作 ----------


class _AlwaysStoppedSafety:
    def check(self):
        return SafetyState(True, False, "人工急停中（按 F12 恢复）")


class _SpyController:
    """记录每次 execute 的动作名。"""

    def __init__(self):
        self.actions: list[str] = []

    def execute(self, action: Action):
        from core.control.base import Result
        self.actions.append(action.name)
        return Result(True, action.name)


def test_run_emergency_stop_blocks_all_actions(tmp_path, monkeypatch):
    from core.contracts import GameState as _GS

    game_cfg = WukongConfig.load("configs/wukong.yaml")
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    spy = _SpyController()

    class _FakeSource:
        def grab(self):
            return frame

    class _FakeAdapter:
        def perceive(self, f):
            return _GS(timestamp=0.0, scene="explore", raw={
                "hp_ratio": 1.0, "enemy_hp_ratio": None, "enemy_present": False,
                "in_combat": False, "gourd_available": True,
                "pose": (0.0, 0.0, 0.0),
                "walkable": {"left": 0.5, "center": 0.8, "right": 0.5, "suggestion": "straight"},
            })

    from core.decision.navigation import CoverageExplorer
    from core.navigation.grid_map import OccupancyGrid
    from games.wukong.combat import CombatDecision

    grid = OccupancyGrid(game_cfg.exploration.grid_size_m, game_cfg.exploration.grid_resolution)
    decision = CombatDecision(game_cfg, CoverageExplorer(grid, game_cfg.exploration), grid)

    monkeypatch.setattr(
        auto_main, "build_wukong",
        lambda path, dry_run=False: (_FakeSource(), _FakeAdapter(), decision, spy, game_cfg),
    )
    monkeypatch.setattr(
        auto_main, "build_safety", lambda cfg, on_release: _AlwaysStoppedSafety()
    )
    settings = Settings(
        runtime=RuntimeConfig(mode="auto", fps=1000.0),
        game=GameRef(name="wukong", window_title="x"),
        recorder=RecorderConfig(output_dir=str(tmp_path)),
        llm=LLMConfig(),
    )
    rc = auto_main.run(settings, "wukong", Path("configs/wukong.yaml"), max_ticks=5)
    assert rc == 0
    # 急停状态下：一切动作被阻断，只有 idle 到达控制器
    assert spy.actions == ["idle"] * 5
    run_dir = next(tmp_path.iterdir())
    first = json.loads((run_dir / "replay.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["output"]["name"] == "idle"
    assert "急停" in first["extra"]["safety"]
    # trace 摘要落盘
    assert (run_dir / "trace_summary.txt").is_file()
