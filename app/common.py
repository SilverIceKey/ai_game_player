"""app 层两个 CLI 入口共享的装配助手（spec §5 顶层运行模式）。

只放两个入口（observe_train / autopilot）都要用的逻辑，避免复制粘贴：
- 配置文件路径解析（settings.yaml 不存在时回退 settings.example.yaml，沿用旧 CLI 风格）
- 配置加载（ConfigError → SystemExit 用户可读报错，不抛 traceback）
- 按 settings.input_device 构建 InputCapture（spec §10）
- session_id 生成（日期时间格式）

平台依赖（pynput / XInput）全部在函数内延迟导入，Linux 开发机可 import 本模块。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from config import ConfigError, GameConfig, Settings, load_game_config, load_settings
from capture.input.base import InputCapture


def resolve_settings_path(config_arg: str, prog: str) -> Path:
    """解析全局配置路径：不存在时回退同目录 settings.example.yaml 并提示。

    沿用旧 apps/auto_player 行为：新克隆仓库只有 example 模板，回退保证可直接启动；
    两者都不存在时 SystemExit 用户可读报错。
    """
    config_path = Path(config_arg)
    if config_path.is_file():
        return config_path
    fallback = config_path.with_name("settings.example.yaml")
    if fallback.is_file():
        print(
            f"[{prog}] {config_path} 不存在，回退使用 {fallback}"
            f"（建议复制为 {config_path} 后按本机环境修改）"
        )
        return fallback
    raise SystemExit(f"配置文件不存在: {config_path}")


def load_configs(
    settings_path: str | Path,
    game: str,
    game_config_arg: str | None,
    prog: str,
) -> tuple[Settings, GameConfig]:
    """加载全局 + 游戏专属配置；ConfigError 转 SystemExit 用户可读报错。"""
    try:
        settings = load_settings(settings_path)
        gc_path = Path(game_config_arg) if game_config_arg else Path(f"configs/{game}.yaml")
        game_config = load_game_config(gc_path, name=game)
    except ConfigError as exc:
        raise SystemExit(f"[{prog}] 配置错误: {exc}") from exc
    return settings, game_config


def build_input_capture(settings: Settings, game_config: GameConfig) -> InputCapture:
    """按 settings.input_device 构建输入采集器（spec §10：统一输出 NormalizedAction）。

    - keyboard_mouse：由 game_config.keys 反推 键位→动作 映射（pynput 延迟到 start()）
    - gamepad：XInput 默认映射（ctypes，构造时触达 win32，非 Windows 抛 RuntimeError）
    """
    if settings.input_device == "gamepad":
        from capture.input.gamepad import GamepadCapture

        return GamepadCapture()

    from capture.input.keyboard_mouse import KeyboardMouseCapture, build_reverse_keymap

    try:
        key_to_action = build_reverse_keymap(game_config.keys)
    except ValueError as exc:
        raise ConfigError(f"keys 键位映射非法: {exc}") from exc
    return KeyboardMouseCapture(key_to_action)


def new_session_id() -> str:
    """session_id：日期时间格式（沿用旧 runs/<timestamp> 命名习惯）。"""
    return datetime.now().strftime("%Y%m%d-%H%M%S")
