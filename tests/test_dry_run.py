"""--dry-run 干跑模式测试：NullController 契约、装配路径、端到端干跑循环。"""
import json
import sys
from pathlib import Path

import numpy as np

from apps.auto_player import main as auto_main
from core.config import GameRef, LLMConfig, RecorderConfig, RuntimeConfig, Settings
from core.contracts import Action
from core.control.base import Controller
from core.control.directinput import DirectInputController
from core.control.null_controller import NullController
from core.decision.navigation import CoverageExplorer
from core.navigation.grid_map import OccupancyGrid
from games.wukong.adapter import WukongAdapter, WukongConfig
from games.wukong.combat import CombatDecision


def test_null_controller_satisfies_protocol():
    controller = NullController()
    assert isinstance(controller, Controller)
    result = controller.execute(Action("light_attack"))
    assert result.success is False
    assert result.detail == "dry-run"
    # 绝不触达真实输入：执行后不得引入 pydirectinput
    assert "pydirectinput" not in sys.modules


def test_build_wukong_dry_run_uses_null_controller():
    _, _, _, controller, cfg = auto_main.build_wukong(Path("configs/wukong.yaml"), dry_run=True)
    assert isinstance(controller, NullController)
    assert cfg.dry_run.frame_interval_ticks > 0
    _, _, _, controller, _ = auto_main.build_wukong(Path("configs/wukong.yaml"), dry_run=False)
    assert isinstance(controller, DirectInputController)


def test_dry_run_loop_end_to_end(tmp_path, monkeypatch):
    """干跑主循环：日志标注 dry_run=true、JSONL 记录 result=dry-run、周期抽样落帧。"""
    cfg = WukongConfig.load("configs/wukong.yaml")
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    class _FakeSource:
        def grab(self):
            return frame

    def _fake_build(game_config_path, dry_run=False):
        adapter = WukongAdapter(cfg)
        grid = OccupancyGrid(cfg.exploration.grid_size_m, cfg.exploration.grid_resolution)
        decision = CombatDecision(cfg, CoverageExplorer(grid, cfg.exploration), grid)
        return _FakeSource(), adapter, decision, NullController(), cfg

    monkeypatch.setattr(auto_main, "build_wukong", _fake_build)
    settings = Settings(
        runtime=RuntimeConfig(mode="auto", fps=1000.0),
        game=GameRef(name="wukong", window_title="x"),
        recorder=RecorderConfig(output_dir=str(tmp_path)),
        llm=LLMConfig(),
    )
    rc = auto_main.run(settings, "wukong", Path("configs/wukong.yaml"),
                       max_ticks=5, dry_run=True)
    assert rc == 0

    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    log_lines = (run_dir / "session.log").read_text(encoding="utf-8").splitlines()
    assert "dry_run=true" in log_lines[0]
    # 逐 tick 三行日志格式不变
    assert any("state scene=" in line for line in log_lines)
    assert any(line.strip().startswith("intent ") for line in log_lines)
    assert any(line.strip().startswith("action ") for line in log_lines)

    records = (run_dir / "session.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(records) == 5
    for line in records:
        assert json.loads(line)["result"] == "dry-run"

    # 无状态转移、无接敌：只有 tick 0 的周期性抽样帧
    frames = list((run_dir / "frames").iterdir())
    assert len(frames) == 1
    assert "sample" in frames[0].name
