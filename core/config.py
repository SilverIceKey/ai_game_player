"""YAML 配置加载与校验。

全局运行配置（configs/settings.yaml）在此校验；
游戏专属配置（configs/<game>.yaml）的 schema 由各游戏适配器自行定义与校验。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """配置缺失或非法。"""


@dataclass(frozen=True)
class RuntimeConfig:
    mode: str = "auto"  # auto=全自动 / assist=半自动建议
    fps: float = 10.0


@dataclass(frozen=True)
class GameRef:
    name: str
    window_title: str
    config_path: str | None = None  # 游戏专属配置路径，缺省为 configs/<name>.yaml


@dataclass(frozen=True)
class RecorderConfig:
    output_dir: str = "runs/"


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "ollama"
    model: str = ""
    vision_model: str = ""  # 复盘图文模型（缺省回退 model）
    base_url: str = ""


@dataclass(frozen=True)
class ReviewConfig:
    """M2 复盘采样与分批参数。"""

    batch_size: int = 4  # 每批送模型的帧数（控制上下文与显存峰值）
    window_ticks: int = 30  # 帧前后各取多少 tick 的操作日志配对
    hp_drop_alert: float = 0.3  # 单 tick 血量跌幅超过该值落异常帧
    sample_interval_ticks: int = 300  # 正式跑周期落帧间隔（300 tick ≈ 30 秒 @10fps）


@dataclass(frozen=True)
class Settings:
    runtime: RuntimeConfig
    game: GameRef
    recorder: RecorderConfig
    llm: LLMConfig
    review: ReviewConfig = ReviewConfig()


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    """读取 YAML 文件并保证顶层是映射。"""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"配置文件不存在: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"配置文件内容必须是 YAML 映射: {p}")
    return data


def require(data: Any, key: str, ctx: str) -> Any:
    """取必填字段，缺失或值为 null 时抛 ConfigError。ctx 为字段路径前缀（如 "game"）。"""
    if not isinstance(data, dict) or data.get(key) is None:
        raise ConfigError(f"配置缺少必填字段: {ctx}.{key}" if ctx else f"配置缺少必填字段: {key}")
    return data[key]


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    sec = data.get(key)
    if sec is None:
        return {}
    if not isinstance(sec, dict):
        raise ConfigError(f"配置字段 {key} 必须是映射，实际为 {type(sec).__name__}")
    return sec


def load_settings(path: str | Path) -> Settings:
    """加载并校验全局运行配置。"""
    data = load_yaml_file(path)

    runtime = _section(data, "runtime")
    mode = runtime.get("mode", "auto")
    if mode not in ("auto", "assist"):
        raise ConfigError(f"runtime.mode 非法: {mode!r}（仅支持 auto/assist）")
    fps = runtime.get("fps", 10.0)
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or fps <= 0:
        raise ConfigError(f"runtime.fps 必须为正数: {fps!r}")

    game = data.get("game")
    if not isinstance(game, dict):
        raise ConfigError("配置缺少必填字段: game")
    game_ref = GameRef(
        name=str(require(game, "name", "game")),
        window_title=str(require(game, "window_title", "game")),
        config_path=game.get("config_path"),
    )

    recorder = _section(data, "recorder")
    llm = _section(data, "llm")
    review = _section(data, "review")
    return Settings(
        runtime=RuntimeConfig(mode=mode, fps=float(fps)),
        game=game_ref,
        recorder=RecorderConfig(output_dir=str(recorder.get("output_dir", "runs/"))),
        llm=LLMConfig(
            provider=str(llm.get("provider", "ollama")),
            model=str(llm.get("model", "")),
            vision_model=str(llm.get("vision_model", "")),
            base_url=str(llm.get("base_url", "")),
        ),
        review=ReviewConfig(
            batch_size=_positive_int(review.get("batch_size", 4), "review.batch_size"),
            window_ticks=_positive_int(review.get("window_ticks", 30), "review.window_ticks"),
            hp_drop_alert=_ratio(review.get("hp_drop_alert", 0.3), "review.hp_drop_alert"),
            sample_interval_ticks=_positive_int(
                review.get("sample_interval_ticks", 300), "review.sample_interval_ticks"
            ),
        ),
    )


def _positive_int(value: Any, ctx: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{ctx} 必须为正整数: {value!r}")
    return value


def _ratio(value: Any, ctx: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not (0.0 < float(value) <= 1.0):
        raise ConfigError(f"{ctx} 必须是 (0, 1] 区间数值: {value!r}")
    return float(value)
