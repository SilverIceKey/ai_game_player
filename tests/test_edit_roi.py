"""--edit-roi 可测逻辑：菜单解析、坐标反缩放、ruamel 配置写回。"""
import shutil
from pathlib import Path

import pytest

from apps.auto_player.edit_roi import (
    EDITABLE_ITEMS,
    parse_menu_choice,
    rect_to_base,
    update_yaml_rect,
)
from core.perception.regions import scale_rect
from games.wukong.adapter import WukongConfig

BASE = (1920, 1080)


def test_parse_menu_choice():
    assert parse_menu_choice("1", 6) == 0
    assert parse_menu_choice(" 3 ", 6) == 2
    assert parse_menu_choice("q", 6) is None
    assert parse_menu_choice("Q", 6) is None
    with pytest.raises(ValueError, match="无效输入"):
        parse_menu_choice("abc", 6)
    with pytest.raises(ValueError, match="超出范围"):
        parse_menu_choice("0", 6)
    with pytest.raises(ValueError, match="超出范围"):
        parse_menu_choice("7", 6)


def test_rect_to_base_inverse_of_scale_rect():
    rect = (140, 1005, 400, 8)
    # 同分辨率：恒等
    assert rect_to_base(rect, (1080, 1920, 3), BASE) == rect
    # 缩放分辨率：scale_rect 后再 rect_to_base 应还原
    frame_shape = (540, 960, 3)
    scaled = scale_rect(rect, frame_shape, BASE)
    assert rect_to_base(scaled, frame_shape, BASE) == pytest.approx(rect, abs=1)


def test_rect_to_base_clamps():
    # 拖框越界时裁剪到基准范围内
    assert rect_to_base((-50, -50, 3000, 3000), (1080, 1920, 3), BASE) == (0, 0, 1920, 1080)


def test_update_yaml_rect_roundtrip(tmp_path):
    src = Path("configs/wukong.yaml")
    target = tmp_path / "wukong.yaml"
    shutil.copy(src, target)

    update_yaml_rect(target, "hp_bar", (150, 1000, 380, 10))
    update_yaml_rect(target, "enemy_search", (500, 180, 900, 520))

    text = target.read_text(encoding="utf-8")
    assert "# 自身血条" in text  # ruamel 往返后注释保留
    assert "# 普通小怪动态血条检测" in text
    cfg = WukongConfig.load(target)  # 值正确且仍通过完整校验
    assert cfg.hud.hp_bar.rect == (150, 1000, 380, 10)
    assert cfg.hud.enemy_search.rect == (500, 180, 900, 520)
    assert cfg.hud.stamina_bar.rect == WukongConfig.load(src).hud.stamina_bar.rect  # 其他项未动


def test_update_yaml_rect_unknown_key(tmp_path):
    target = tmp_path / "wukong.yaml"
    shutil.copy("configs/wukong.yaml", target)
    with pytest.raises(KeyError, match="hud.nope"):
        update_yaml_rect(target, "nope", (0, 0, 10, 10))


def test_editable_items_match_yaml():
    """菜单项必须与 configs/wukong.yaml 的 hud 段字段一一对应（防菜单漂移）。"""
    import yaml
    doc = yaml.safe_load(Path("configs/wukong.yaml").read_text(encoding="utf-8"))
    for key, _label in EDITABLE_ITEMS:
        assert key in doc["hud"], f"菜单项 {key} 不在 hud 段"
        assert "rect" in doc["hud"][key], f"hud.{key} 缺少 rect（不可拖框校准）"
