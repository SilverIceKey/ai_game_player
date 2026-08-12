"""YAML 配置加载与校验（SPEC v1.0 新 schema）。

- 全局运行配置：`configs/settings.yaml`（采集/模型/预测/标签/采样/损失权重，spec §13/§23/§28）
- 游戏专属配置：`configs/<game>.yaml`（窗口、键位映射 §9、安全参数 §39）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from capture.action import BUTTONS


class ConfigError(ValueError):
    """配置缺失或非法。"""


# ---------- 全局运行配置 ----------


@dataclass(frozen=True)
class CaptureConfig:
    source_fps: float = 60.0  # spec §13：采集帧率


@dataclass(frozen=True)
class ModelConfig:
    sample_fps: float = 12.0  # spec §13：模型时间窗采样率
    history_frames: int = 16  # spec §13/§8.1：Video History 窗口
    input_width: int = 384  # spec §14
    input_height: int = 216  # spec §14
    history_actions: int = 16  # spec §8.2：Action History 窗口


@dataclass(frozen=True)
class PredictionConfig:
    action_step_ms: float = 50.0  # spec §13/§15
    future_action_steps: int = 4  # spec §13/§15


@dataclass(frozen=True)
class LabelsConfig:
    action_label_offset_ms: float = 0.0  # spec §12：人类反应延迟补偿，实验搜索 0~250


@dataclass(frozen=True)
class SamplingConfig:
    """Replay Buffer 采样权重（spec §28，初始值仅为搜索起点）。"""

    historical: float = 0.50
    recent: float = 0.25
    correction: float = 0.20
    rare: float = 0.05


@dataclass(frozen=True)
class LossWeights:
    """spec §23：所有 Loss 权重必须配置化。"""

    move: float = 1.0
    camera: float = 1.0
    button: float = 1.0
    temporal: float = 1.0


@dataclass(frozen=True)
class Settings:
    game: str
    sessions_dir: str = "sessions/"
    dataset_version: str = "dataset-v001"
    input_device: str = "keyboard_mouse"  # keyboard_mouse / gamepad（spec §10）
    capture: CaptureConfig = CaptureConfig()
    model: ModelConfig = ModelConfig()
    prediction: PredictionConfig = PredictionConfig()
    labels: LabelsConfig = LabelsConfig()
    sampling: SamplingConfig = SamplingConfig()
    loss_weights: LossWeights = LossWeights()


# ---------- 游戏专属配置 ----------


@dataclass(frozen=True)
class WindowConfig:
    title: str
    capture_backend: str = "auto"  # auto / mss / wgc
    foreground_on_start: bool = True
    rect: tuple[int, int, int, int] | None = None  # 手动截屏区域（仅 mss）


@dataclass(frozen=True)
class SafetyConfig:
    """spec §39 Safety Filter / §26 Human Override 参数。"""

    override_key: str = "F12"  # §26 人工接管键（toggle）
    episode_key: str = "F9"  # §21 手动 START/STOP EPISODE
    stop_on_focus_lost: bool = True  # §39：失焦立即 STOP ACTION
    max_button_hold_ms: float = 5000.0  # §39：最大连续按键时间
    max_camera_delta: float = 0.5  # §39：单步最大视角量（归一化）
    max_action_rate_hz: float = 40.0  # §39：最大动作频率
    inference_timeout_ms: float = 100.0  # §47：模型超时 → Pause AI + Release Input


@dataclass(frozen=True)
class ExecutorConfig:
    pixels_per_unit: float = 400.0  # 归一化 camera 1.0 对应鼠标像素（实机校准）
    action_pause: float = 0.01  # 输入后间隔，避免输入洪峰
    move_deadzone: float = 0.15  # move 轴死区


@dataclass(frozen=True)
class GameConfig:
    name: str
    window: WindowConfig
    keys: dict[str, str] = field(default_factory=dict)  # §9 动作 → 实际键位（mouse_* 前缀为鼠标键）
    safety: SafetyConfig = SafetyConfig()
    executor: ExecutorConfig = ExecutorConfig()


# ---------- 加载与校验 ----------


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"配置文件不存在: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"配置文件内容必须是 YAML 映射: {p}")
    return data


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    sec = data.get(key)
    if sec is None:
        return {}
    if not isinstance(sec, dict):
        raise ConfigError(f"配置字段 {key} 必须是映射，实际为 {type(sec).__name__}")
    return sec


def _positive_float(value: Any, ctx: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0:
        raise ConfigError(f"{ctx} 必须为正数: {value!r}")
    return float(value)


def _positive_int(value: Any, ctx: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{ctx} 必须为正整数: {value!r}")
    return value


def _non_negative_float(value: Any, ctx: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) < 0:
        raise ConfigError(f"{ctx} 必须为非负数: {value!r}")
    return float(value)


def load_settings(path: str | Path) -> Settings:
    data = load_yaml_file(path)
    game = data.get("game")
    if not isinstance(game, str) or not game.strip():
        raise ConfigError("配置缺少必填字段: game（游戏名，对应 configs/<game>.yaml）")

    device = data.get("input_device", "keyboard_mouse")
    if device not in ("keyboard_mouse", "gamepad"):
        raise ConfigError(f"input_device 非法: {device!r}（仅支持 keyboard_mouse/gamepad）")

    capture = _section(data, "capture")
    model = _section(data, "model")
    prediction = _section(data, "prediction")
    labels = _section(data, "labels")
    sampling = _section(data, "sampling")
    loss = _section(data, "loss_weights")

    return Settings(
        game=game.strip(),
        sessions_dir=str(data.get("sessions_dir", "sessions/")),
        dataset_version=str(data.get("dataset_version", "dataset-v001")),
        input_device=device,
        capture=CaptureConfig(
            source_fps=_positive_float(capture.get("source_fps", 60.0), "capture.source_fps"),
        ),
        model=ModelConfig(
            sample_fps=_positive_float(model.get("sample_fps", 12.0), "model.sample_fps"),
            history_frames=_positive_int(model.get("history_frames", 16), "model.history_frames"),
            input_width=_positive_int(model.get("input_width", 384), "model.input_width"),
            input_height=_positive_int(model.get("input_height", 216), "model.input_height"),
            history_actions=_positive_int(
                model.get("history_actions", 16), "model.history_actions"
            ),
        ),
        prediction=PredictionConfig(
            action_step_ms=_positive_float(
                prediction.get("action_step_ms", 50.0), "prediction.action_step_ms"
            ),
            future_action_steps=_positive_int(
                prediction.get("future_action_steps", 4), "prediction.future_action_steps"
            ),
        ),
        labels=LabelsConfig(
            action_label_offset_ms=_non_negative_float(
                labels.get("action_label_offset_ms", 0.0), "labels.action_label_offset_ms"
            ),
        ),
        sampling=SamplingConfig(
            historical=_non_negative_float(sampling.get("historical", 0.5), "sampling.historical"),
            recent=_non_negative_float(sampling.get("recent", 0.25), "sampling.recent"),
            correction=_non_negative_float(
                sampling.get("correction", 0.2), "sampling.correction"
            ),
            rare=_non_negative_float(sampling.get("rare", 0.05), "sampling.rare"),
        ),
        loss_weights=LossWeights(
            move=_non_negative_float(loss.get("move", 1.0), "loss_weights.move"),
            camera=_non_negative_float(loss.get("camera", 1.0), "loss_weights.camera"),
            button=_non_negative_float(loss.get("button", 1.0), "loss_weights.button"),
            temporal=_non_negative_float(loss.get("temporal", 1.0), "loss_weights.temporal"),
        ),
    )


def load_game_config(path: str | Path, name: str = "") -> GameConfig:
    data = load_yaml_file(path)

    window = _section(data, "window")
    title = window.get("title")
    if not isinstance(title, str) or not title.strip():
        raise ConfigError("配置缺少必填字段: window.title（用窗口枚举工具查看实际标题）")
    backend = window.get("capture_backend", "auto")
    if backend not in ("auto", "mss", "wgc"):
        raise ConfigError(f"window.capture_backend 非法: {backend!r}（仅支持 auto/mss/wgc）")
    rect = window.get("rect")
    if rect is not None:
        if not isinstance(rect, list) or len(rect) != 4 or not all(isinstance(v, int) for v in rect):
            raise ConfigError(f"window.rect 必须是 [x, y, w, h] 四整数列表: {rect!r}")
        rect = tuple(rect)

    keys = _section(data, "keys")
    keymap: dict[str, str] = {}
    for action_name, key in keys.items():
        if action_name not in BUTTONS and not action_name.startswith("move_"):
            raise ConfigError(
                f"keys.{action_name} 不是合法动作名（合法值: {BUTTONS} + move_forward/back/left/right）"
            )
        if not isinstance(key, str) or not key.strip():
            raise ConfigError(f"keys.{action_name} 键位必须为非空字符串: {key!r}")
        keymap[action_name] = key.strip()

    safety = _section(data, "safety")
    executor = _section(data, "executor")

    return GameConfig(
        name=name or Path(path).stem,
        window=WindowConfig(
            title=title.strip(),
            capture_backend=backend,
            foreground_on_start=bool(window.get("foreground_on_start", True)),
            rect=rect,
        ),
        keys=keymap,
        safety=SafetyConfig(
            override_key=str(safety.get("override_key", "F12")),
            episode_key=str(safety.get("episode_key", "F9")),
            stop_on_focus_lost=bool(safety.get("stop_on_focus_lost", True)),
            max_button_hold_ms=_positive_float(
                safety.get("max_button_hold_ms", 5000.0), "safety.max_button_hold_ms"
            ),
            max_camera_delta=_positive_float(
                safety.get("max_camera_delta", 0.5), "safety.max_camera_delta"
            ),
            max_action_rate_hz=_positive_float(
                safety.get("max_action_rate_hz", 40.0), "safety.max_action_rate_hz"
            ),
            inference_timeout_ms=_positive_float(
                safety.get("inference_timeout_ms", 100.0), "safety.inference_timeout_ms"
            ),
        ),
        executor=ExecutorConfig(
            pixels_per_unit=_positive_float(
                executor.get("pixels_per_unit", 400.0), "executor.pixels_per_unit"
            ),
            action_pause=_non_negative_float(
                executor.get("action_pause", 0.01), "executor.action_pause"
            ),
            move_deadzone=_non_negative_float(
                executor.get("move_deadzone", 0.15), "executor.move_deadzone"
            ),
        ),
    )
