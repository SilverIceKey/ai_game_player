"""配置加载与校验测试：合法配置解析、缺字段/非法值报错。"""
from pathlib import Path

import pytest

from core.config import ConfigError, load_settings
from games.wukong.adapter import WukongConfig

VALID = """
runtime:
  mode: auto
  fps: 10
game:
  name: wukong
  window_title: "黑神话：悟空"
recorder:
  output_dir: "runs/"
"""


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "settings.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_load_valid_settings(tmp_path):
    settings = load_settings(_write(tmp_path, VALID))
    assert settings.runtime.mode == "auto"
    assert settings.runtime.fps == 10.0
    assert settings.game.name == "wukong"
    assert settings.game.window_title == "黑神话：悟空"
    assert settings.recorder.output_dir == "runs/"


def test_missing_window_title_raises(tmp_path):
    bad = "game:\n  name: wukong\n"
    with pytest.raises(ConfigError, match="game.window_title"):
        load_settings(_write(tmp_path, bad))


def test_missing_game_section_raises(tmp_path):
    with pytest.raises(ConfigError, match="game"):
        load_settings(_write(tmp_path, "runtime:\n  mode: auto\n"))


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="不存在"):
        load_settings(tmp_path / "nope.yaml")


def test_invalid_mode_raises(tmp_path):
    bad = VALID.replace("mode: auto", "mode: turbo")
    with pytest.raises(ConfigError, match="runtime.mode"):
        load_settings(_write(tmp_path, bad))


def test_invalid_fps_raises(tmp_path):
    bad = VALID.replace("fps: 10", "fps: 0")
    with pytest.raises(ConfigError, match="runtime.fps"):
        load_settings(_write(tmp_path, bad))


def test_load_repo_example_settings():
    settings = load_settings("configs/settings.example.yaml")
    assert settings.game.name == "wukong"
    assert settings.runtime.fps > 0


def test_load_repo_wukong_yaml():
    cfg = WukongConfig.load("configs/wukong.yaml")
    assert cfg.window.title
    assert cfg.perception.base_resolution == (1920, 1080)
    assert cfg.hud.hp_bar.rect[2] > 0
    for key in ("move_forward", "light_attack", "dodge", "heal", "lock_on"):
        assert cfg.keys[key]
    assert 0.0 < cfg.combat.heal_hp_threshold < 1.0


def test_wukong_config_missing_hud_raises():
    with pytest.raises(ConfigError, match="hud"):
        WukongConfig.from_dict({"window": {"title": "x"}, "keys": {}})


def test_wukong_config_missing_key_raises():
    cfg = WukongConfig.load("configs/wukong.yaml")
    import yaml as _yaml
    data = _yaml.safe_load(Path("configs/wukong.yaml").read_text(encoding="utf-8"))
    del data["keys"]["dodge"]
    with pytest.raises(ConfigError, match="keys.dodge"):
        WukongConfig.from_dict(data)
    assert cfg.keys["dodge"]  # 原配置未被污染


def test_wukong_config_unknown_field_raises():
    import yaml as _yaml
    data = _yaml.safe_load(Path("configs/wukong.yaml").read_text(encoding="utf-8"))
    data["combat"]["heal_hp_threashold"] = 0.3  # 笔误字段必须报错
    with pytest.raises(ConfigError, match="未知字段"):
        WukongConfig.from_dict(data)
