"""全自动模式主循环装配 + 逐 tick 日志（计划文档第 2 节主链路、3.4 节日志格式）。

主链路：截屏 → perceive → 战斗 FSM/探索决策 → 模拟键鼠 → JSONL 记录 → 逐 tick 日志。
日志双写：控制台 + runs/<timestamp>/session.log；状态转移与首次接敌落关键帧截图。
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path

from core.config import Settings, load_settings
from core.contracts import Action, GameState
from core.recorder.base import StepRecord
from core.recorder.jsonl import JsonlRecorder

_LOG_INDENT = " " * 15  # 与 "[HH:MM:SS.mmm] " 等宽，续行对齐


def format_tick(state: GameState, intent: str, action: Action) -> str:
    """逐 tick 日志（计划文档 3.4 节）：state / intent / action 三行。"""
    raw = state.raw
    enemy = raw.get("enemy_hp_ratio")
    enemy_s = f"{float(enemy):.2f}" if isinstance(enemy, (int, float)) else "-"
    pose = raw.get("pose") or (0.0, 0.0, 0.0)
    degrees = int(round(math.degrees(float(pose[2]))))
    params = " ".join(f"{k}={v}" for k, v in action.params.items())
    lines = [
        f"state scene={state.scene} "
        f"hp={float(raw.get('hp_ratio') or 0.0):.2f} "
        f"stamina={float(raw.get('stamina_ratio') or 0.0):.2f} "
        f"enemy_hp={enemy_s} "
        f"gourd={1 if raw.get('gourd_available') else 0} "
        f"pos=({float(pose[0]):.1f},{float(pose[1]):.1f},{degrees}°)",
        f"intent {intent}",
        f"action {action.name}" + (f" {params}" if params else ""),
    ]
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    return f"[{timestamp}] " + f"\n{_LOG_INDENT}".join(lines)


def setup_logger(log_path: Path) -> logging.Logger:
    """控制台 + 文件双写；消息内自带时间戳，formatter 原样输出。"""
    logger = logging.getLogger("auto_player")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        formatter = logging.Formatter("%(message)s")
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def build_wukong(game_config_path: Path):
    """装配悟空链路各组件（导入延迟到选择游戏时，避免无关适配器依赖）。"""
    from core.control.directinput import DirectInputController
    from core.decision.navigation import CoverageExplorer
    from core.navigation.grid_map import OccupancyGrid
    from core.perception.mss_source import WindowFrameSource
    from games.wukong.adapter import WukongAdapter, WukongConfig
    from games.wukong.combat import CombatDecision

    config = WukongConfig.load(game_config_path)
    adapter = WukongAdapter(config)
    grid = OccupancyGrid(config.exploration.grid_size_m, config.exploration.grid_resolution)
    explorer = CoverageExplorer(grid, config.exploration)
    decision = CombatDecision(config, explorer, grid)
    controller = DirectInputController(config.keys, config.control)
    source = WindowFrameSource(config.window.title, rect=config.window.rect)
    return source, adapter, decision, controller


def run(settings: Settings, game: str, game_config_path: Path, max_ticks: int = 0) -> int:
    run_dir = Path(settings.recorder.output_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    recorder = JsonlRecorder(run_dir)
    logger = setup_logger(run_dir / "session.log")

    if game != "wukong":
        raise SystemExit(f"未知游戏适配器: {game}（M1 仅支持 wukong）")
    source, adapter, decision, controller = build_wukong(game_config_path)

    if settings.runtime.mode != "auto":
        logger.warning("[%s] runtime.mode=%s，auto_player 为全自动模式，建议 mode=auto",
                       datetime.now().strftime("%H:%M:%S.%f")[:-3], settings.runtime.mode)

    interval = 1.0 / settings.runtime.fps
    logger.info("[%s] session start game=%s fps=%g run_dir=%s",
                datetime.now().strftime("%H:%M:%S.%f")[:-3], game, settings.runtime.fps, run_dir)

    tick = 0
    prev_fsm = decision.state_name
    saved_first_combat = False
    try:
        while True:
            started = time.monotonic()
            frame = source.grab()
            state = adapter.perceive(frame)
            action = decision.decide(state)
            result = controller.execute(action)
            recorder.record(StepRecord(
                timestamp=state.timestamp, state=state, output=action, result=result.detail,
            ))
            logger.info(format_tick(state, decision.intent, action))

            # 状态转移与首次接敌落关键帧截图（计划文档 3.4 节）
            transitioned = decision.state_name != prev_fsm
            first_combat = bool(state.raw.get("in_combat")) and not saved_first_combat
            if transitioned or first_combat:
                recorder.save_frame(frame, f"{tick:06d}_{decision.state_name}")
                prev_fsm = decision.state_name
                if first_combat:
                    saved_first_combat = True

            tick += 1
            if max_ticks > 0 and tick >= max_ticks:
                break
            elapsed = time.monotonic() - started
            if elapsed < interval:
                time.sleep(interval - elapsed)
    except KeyboardInterrupt:
        logger.info("[%s] interrupted by user at tick %d",
                    datetime.now().strftime("%H:%M:%S.%f")[:-3], tick)
    finally:
        replay = recorder.export()
        recorder.close()
        logger.info("[%s] session end ticks=%d replay=%s",
                    datetime.now().strftime("%H:%M:%S.%f")[:-3], tick, replay)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="全自动游戏 AI（仅截屏 + 模拟输入，不读内存不注入）"
    )
    parser.add_argument("--game", required=True, help="游戏适配器名，如 wukong")
    parser.add_argument("--config", default="configs/settings.yaml", help="全局配置文件路径")
    parser.add_argument("--game-config", default=None,
                        help="游戏专属配置路径，缺省 configs/<game>.yaml")
    parser.add_argument("--max-ticks", type=int, default=0,
                        help="最多执行 tick 数（0=不限，调试用）")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_file():
        # 新克隆仓库只有 settings.example.yaml：回退并提示，保证 F5 可直接启动
        fallback = config_path.with_name("settings.example.yaml")
        if fallback.is_file():
            print(f"[auto_player] {config_path} 不存在，回退使用 {fallback}"
                  f"（建议复制为 {config_path} 后按本机环境修改）")
            config_path = fallback
        else:
            raise SystemExit(f"配置文件不存在: {config_path}")
    settings = load_settings(config_path)

    game_config = Path(args.game_config) if args.game_config else Path(f"configs/{args.game}.yaml")
    return run(settings, args.game, game_config, max_ticks=args.max_ticks)


if __name__ == "__main__":
    sys.exit(main())
