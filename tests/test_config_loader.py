"""config.py 加载与校验测试（SPEC v1.0 新 schema）。

覆盖：合法配置加载、缺字段/非法值的明确 ConfigError、
仓库真实 configs/settings.example.yaml 与 configs/wukong.yaml 加载通过（防 schema 漂移）。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from config import ConfigError, load_game_config, load_settings, load_yaml_file

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_YAML = REPO_ROOT / "configs" / "settings.example.yaml"
WUKONG_YAML = REPO_ROOT / "configs" / "wukong.yaml"


def _write(tmp_path: Path, text: str, name: str = "settings.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# ---------- settings 加载 ----------


def test_load_settings_minimal_valid(tmp_path: Path) -> None:
    path = _write(tmp_path, "game: wukong\n")
    settings = load_settings(path)
    assert settings.game == "wukong"
    # 缺省段全部落 spec §13/§28/§23 默认值
    assert settings.capture.source_fps == 60.0
    assert settings.model.sample_fps == 12.0
    assert settings.model.history_frames == 16
    assert settings.model.input_width == 384
    assert settings.model.input_height == 216
    assert settings.prediction.action_step_ms == 50.0
    assert settings.prediction.future_action_steps == 4
    assert settings.sampling.historical == 0.5
    assert settings.sampling.correction == 0.2
    assert settings.loss_weights.temporal == 1.0


def test_load_settings_full_valid(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
game: wukong
sessions_dir: "data/sessions/"
dataset_version: "dataset-v007"
input_device: gamepad
capture: {source_fps: 30}
model: {sample_fps: 10, history_frames: 8, input_width: 320, input_height: 180}
prediction: {action_step_ms: 40, future_action_steps: 5}
labels: {action_label_offset_ms: 120}
sampling: {historical: 0.4, recent: 0.3, correction: 0.2, rare: 0.1}
loss_weights: {move: 1.0, camera: 2.0, button: 1.5, temporal: 0.5}
""",
    )
    settings = load_settings(path)
    assert settings.input_device == "gamepad"
    assert settings.dataset_version == "dataset-v007"
    assert settings.capture.source_fps == 30.0
    assert settings.model.history_frames == 8
    assert settings.labels.action_label_offset_ms == 120.0
    assert settings.loss_weights.camera == 2.0


def test_load_settings_missing_game(tmp_path: Path) -> None:
    path = _write(tmp_path, "capture: {source_fps: 60}\n")
    with pytest.raises(ConfigError, match="game"):
        load_settings(path)


def test_load_settings_invalid_input_device(tmp_path: Path) -> None:
    path = _write(tmp_path, "game: wukong\ninput_device: steering_wheel\n")
    with pytest.raises(ConfigError, match="input_device"):
        load_settings(path)


def test_load_settings_non_positive_fps(tmp_path: Path) -> None:
    path = _write(tmp_path, "game: wukong\ncapture: {source_fps: 0}\n")
    with pytest.raises(ConfigError, match="source_fps"):
        load_settings(path)


def test_load_settings_wrong_type(tmp_path: Path) -> None:
    path = _write(tmp_path, "game: wukong\nmodel: {history_frames: 3.5}\n")
    with pytest.raises(ConfigError, match="history_frames"):
        load_settings(path)


def test_load_yaml_file_missing(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="不存在"):
        load_yaml_file(tmp_path / "nope.yaml")


def test_load_yaml_file_not_mapping(tmp_path: Path) -> None:
    path = _write(tmp_path, "- just\n- a\n- list\n")
    with pytest.raises(ConfigError, match="映射"):
        load_yaml_file(path)


# ---------- game config 加载 ----------


def test_load_game_config_valid(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
window: {title: "Test Game", capture_backend: mss}
keys: {move_forward: "w", attack_light: "mouse_left"}
safety: {override_key: "F12", episode_key: "F9"}
""",
        name="test.yaml",
    )
    gc = load_game_config(path)
    assert gc.name == "test"
    assert gc.window.title == "Test Game"
    assert gc.window.capture_backend == "mss"
    assert gc.keys == {"move_forward": "w", "attack_light": "mouse_left"}
    assert gc.safety.override_key == "F12"
    assert gc.safety.inference_timeout_ms == 100.0


def test_load_game_config_missing_title(tmp_path: Path) -> None:
    path = _write(tmp_path, "window: {capture_backend: mss}\n", name="g.yaml")
    with pytest.raises(ConfigError, match="window.title"):
        load_game_config(path)


def test_load_game_config_invalid_backend(tmp_path: Path) -> None:
    path = _write(tmp_path, 'window: {title: "t", capture_backend: dxgi}\n', name="g.yaml")
    with pytest.raises(ConfigError, match="capture_backend"):
        load_game_config(path)


def test_load_game_config_invalid_key_action(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        'window: {title: "t"}\nkeys: {fireball: "q"}\n',
        name="g.yaml",
    )
    with pytest.raises(ConfigError, match="fireball"):
        load_game_config(path)


def test_load_game_config_invalid_rect(tmp_path: Path) -> None:
    path = _write(
        tmp_path, 'window: {title: "t", rect: [0, 0, 100]}\n', name="g.yaml"
    )
    with pytest.raises(ConfigError, match="rect"):
        load_game_config(path)


# ---------- voice 段（PLAN-20260814-autopilot-voice-v1） ----------


def test_load_settings_voice_defaults(tmp_path: Path) -> None:
    settings = load_settings(_write(tmp_path, "game: wukong\n"))
    assert settings.voice.enabled is False
    assert settings.voice.addr == "192.168.5.249:18103"
    assert settings.voice.decision_interval_s == 5.0


def test_load_settings_voice_full(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
game: wukong
voice:
  enabled: true
  addr: "10.0.0.2:9000"
  speed: 1.2
  language: "ZH"
  speaker: "default"
  decision_interval_s: 0
""",
    )
    voice = load_settings(path).voice
    assert voice.enabled is True
    assert voice.addr == "10.0.0.2:9000"
    assert voice.speed == 1.2
    assert voice.language == "ZH"
    assert voice.decision_interval_s == 0.0


def test_load_settings_voice_enabled_requires_addr(tmp_path: Path) -> None:
    path = _write(tmp_path, 'game: wukong\nvoice: {enabled: true, addr: ""}\n')
    with pytest.raises(ConfigError, match="voice.addr"):
        load_settings(path)


# ---------- 仓库真实配置（防 schema 漂移） ----------


def test_repo_example_yaml_loads() -> None:
    settings = load_settings(EXAMPLE_YAML)
    assert settings.game == "wukong"
    assert settings.capture.source_fps == 60.0
    assert settings.prediction.future_action_steps == 4
    total = (
        settings.sampling.historical
        + settings.sampling.recent
        + settings.sampling.correction
        + settings.sampling.rare
    )
    assert total == pytest.approx(1.0)


def test_repo_wukong_yaml_loads() -> None:
    gc = load_game_config(WUKONG_YAML)
    assert gc.window.title.strip()
    # 旧配置语义沿用：WASD 移动、mouse_left 轻棍、space 闪避、r 喝酒、mouse_middle 锁定
    assert gc.keys["move_forward"] == "w"
    assert gc.keys["attack_light"] == "mouse_left"
    assert gc.keys["dodge"] == "space"
    assert gc.keys["heal"] == "r"
    assert gc.keys["lock_target"] == "mouse_middle"
    assert gc.safety.episode_key == "F9"
    assert gc.safety.override_key == "F12"
