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


def build_wukong(game_config_path: Path, dry_run: bool = False):
    """装配悟空链路各组件（导入延迟到选择游戏时，避免无关适配器依赖）。

    dry_run=True 时使用 NullController：链路照常跑，绝不触达真实输入。
    """
    from core.control.directinput import DirectInputController
    from core.control.null_controller import NullController
    from core.decision.navigation import CoverageExplorer
    from core.navigation.grid_map import OccupancyGrid
    from core.perception.source_factory import build_frame_source
    from games.wukong.adapter import WukongAdapter, WukongConfig
    from games.wukong.combat import CombatDecision

    config = WukongConfig.load(game_config_path)
    adapter = WukongAdapter(config)
    grid = OccupancyGrid(config.exploration.grid_size_m, config.exploration.grid_resolution)
    explorer = CoverageExplorer(grid, config.exploration)
    decision = CombatDecision(config, explorer, grid)
    controller = NullController() if dry_run else DirectInputController(config.keys, config.control)
    source = build_frame_source(config.window)
    return source, adapter, decision, controller, config


def run(settings: Settings, game: str, game_config_path: Path, max_ticks: int = 0,
        dry_run: bool = False) -> int:
    run_dir = Path(settings.recorder.output_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    recorder = JsonlRecorder(run_dir)
    logger = setup_logger(run_dir / "session.log")

    if game != "wukong":
        raise SystemExit(f"未知游戏适配器: {game}（M1 仅支持 wukong）")
    source, adapter, decision, controller, game_config = build_wukong(game_config_path, dry_run)

    if settings.runtime.mode != "auto":
        logger.warning("[%s] runtime.mode=%s，auto_player 为全自动模式，建议 mode=auto",
                       datetime.now().strftime("%H:%M:%S.%f")[:-3], settings.runtime.mode)

    interval = 1.0 / settings.runtime.fps
    logger.info("[%s] session start game=%s fps=%g dry_run=%s run_dir=%s",
                datetime.now().strftime("%H:%M:%S.%f")[:-3], game, settings.runtime.fps,
                "true" if dry_run else "false", run_dir)

    # 正式跑（非 dry-run）启动时把游戏窗口提前台（mss 屏幕区域截屏必需；
    # dry-run / calibrate 不发输入，没必要抢焦点）
    if not dry_run and game_config.window.foreground_on_start:
        from core.perception.foreground import bring_to_foreground

        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        if bring_to_foreground(game_config.window.title):
            logger.info("[%s] foreground: 游戏窗口已提前台", ts)
        else:
            logger.info("[%s] foreground: 未能把游戏窗口提前台"
                        "（非 Windows 或未找到窗口/被系统拦截）；"
                        "使用 mss 后端时请手动切到游戏窗口", ts)

    tick = 0
    prev_fsm = decision.state_name
    saved_first_combat = False
    frame_interval = game_config.dry_run.frame_interval_ticks
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
            elif dry_run and tick % frame_interval == 0:
                # 干跑模式：周期性抽样落帧，供用户核对 HUD 区域与阈值
                recorder.save_frame(frame, f"{tick:06d}_sample")

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
    parser.add_argument("--game", default=None, help="游戏适配器名，如 wukong")
    parser.add_argument("--list-windows", action="store_true",
                        help="列出当前所有可见窗口标题与区域后退出"
                             "（校准 configs/<game>.yaml 的 window.title 用，无需 --game）")
    parser.add_argument("--config", default="configs/settings.yaml", help="全局配置文件路径")
    parser.add_argument("--game-config", default=None,
                        help="游戏专属配置路径，缺省 configs/<game>.yaml")
    parser.add_argument("--max-ticks", type=int, default=0,
                        help="最多执行 tick 数（0=不限，调试用）")
    parser.add_argument("--dry-run", action="store_true",
                        help="干跑模式：完整跑截屏-感知-决策-日志-记录链路，"
                             "但不发任何键鼠输入（首次实机核对配置用）")
    parser.add_argument("--calibrate", action="store_true",
                        help="HUD 校准：抓一帧，输出整图标注/区域裁剪/测量值到 "
                             "runs/<timestamp>/calib/ 后退出（不发输入）")
    parser.add_argument("--probe-input", action="store_true",
                        help="输入链路诊断：倒计时后逐个动作发送真实输入并播报"
                             "（确认哪些动作在游戏内实际生效，定位输入问题）")
    args = parser.parse_args(argv)

    if args.list_windows:
        from core.perception.mss_source import list_visible_windows

        try:
            windows = list_visible_windows()
        except RuntimeError as exc:
            raise SystemExit(f"[auto_player] {exc}") from exc
        for title, rect in windows:
            print(f"{title!r}\t{rect}")
        return 0

    if not args.game:
        parser.error("缺少 --game（--list-windows 模式除外）")

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
    try:
        if args.calibrate:
            return run_calibrate_cli(settings, args.game, game_config)
        if args.probe_input:
            return run_probe_cli(args.game, game_config)
        return run(settings, args.game, game_config, max_ticks=args.max_ticks, dry_run=args.dry_run)
    except RuntimeError as exc:
        # 截屏后端/窗口定位等启动失败：明确报错，不给 traceback
        raise SystemExit(f"[auto_player] 启动失败: {exc}") from exc


def run_probe_cli(game: str, game_config_path: Path) -> int:
    """--probe-input 入口：逐动作发送真实输入，用户对照游戏画面确认生效情况。"""
    if game != "wukong":
        raise SystemExit(f"未知游戏适配器: {game}（M1 仅支持 wukong）")
    from apps.auto_player.probe import run_probe
    from core.control.directinput import DirectInputController
    from core.perception.foreground import bring_to_foreground
    from games.wukong.adapter import WukongConfig

    config = WukongConfig.load(game_config_path)
    foregrounded = bring_to_foreground(config.window.title)
    print(f"[probe] 游戏窗口提前台: {'成功' if foregrounded else '失败（倒计时内请手动切换到游戏）'}")
    controller = DirectInputController(config.keys, config.control)
    run_probe(controller)
    return 0


def run_calibrate_cli(settings: Settings, game: str, game_config_path: Path) -> int:
    """--calibrate 入口：抓帧失败时给出明确报错而非 traceback。"""
    if game != "wukong":
        raise SystemExit(f"未知游戏适配器: {game}（M1 仅支持 wukong）")
    from apps.auto_player.calibrate import run_calibrate
    from core.perception.source_factory import build_frame_source
    from games.wukong.adapter import WukongConfig

    config = WukongConfig.load(game_config_path)
    try:
        source = build_frame_source(config.window)
        out_dir = run_calibrate(config, source, settings.recorder.output_dir)
    except Exception as exc:
        raise SystemExit(
            f"[calibrate] 抓帧失败: {exc}\n"
            "请确认游戏已启动、窗口标题与 configs/wukong.yaml 的 window.title 一致"
            "（或配置 window.rect 手动指定截屏区域）；"
            "Linux 开发机无法抓屏，--calibrate 仅支持 Windows 实机。"
        ) from exc
    print(f"[calibrate] 完成，输出目录: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
