"""HUD 校准模式（--calibrate）。

首次实机校准工作流（计划文档第 5 节假设 3）：
启动游戏到战斗/探索画面 → 跑 --calibrate → 看 calib/ 里的标注图与裁剪图 →
区域套错了就改 configs/wukong.yaml 坐标 → 再跑一遍确认。

抓一帧输出到 runs/<timestamp>/calib/：
- annotated.png：整图 + 所有 HUD 区域矩形框与名称标签（bar=黄，presence=紫）
- <区域名>.png：每个区域的单独裁剪
- 控制台打印 perceive 用到的原始测量值
抓完即退出，不发任何键鼠输入。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from core.perception.bars import detect_bar
from core.perception.base import FrameSource
from core.perception.regions import (
    BarSpec,
    PresenceSpec,
    crop_region,
    detect_presence,
    match_ratio,
    measure_bar,
    scale_rect,
)
from games.wukong.adapter import WukongConfig

BAR_COLOR = (0, 255, 255)  # BGR 黄：条状区域（血条/体力条/Boss 固定条）
PRESENCE_COLOR = (255, 0, 255)  # BGR 紫：块状区域（葫芦/死亡提示）
SEARCH_COLOR = (255, 255, 0)  # BGR 青：动态小怪血条搜索区域


def _region_items(config: WukongConfig) -> list[tuple[str, BarSpec | PresenceSpec]]:
    hud = config.hud
    return [
        ("hp_bar", hud.hp_bar),
        ("stamina_bar", hud.stamina_bar),
        ("mp_bar", hud.mp_bar),
        ("enemy_hp_bar", hud.enemy_hp_bar),
        ("gourd", hud.gourd),
        ("dead_indicator", hud.dead_indicator),
    ]


def annotate_frame(frame: np.ndarray, config: WukongConfig) -> np.ndarray:
    """整图标注：固定 ROI 画矩形框 + 名称标签；enemy_search 搜索区域用青色区分。"""
    annotated = frame.copy()
    base = config.perception.base_resolution
    for name, spec in _region_items(config):
        x, y, w, h = scale_rect(spec.rect, frame.shape, base)
        color = BAR_COLOR if isinstance(spec, BarSpec) else PRESENCE_COLOR
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            annotated, name, (x, max(12, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
    # 动态小怪血条搜索区域（区别于固定 ROI 的框色）
    x, y, w, h = scale_rect(config.hud.enemy_search.rect, frame.shape, base)
    cv2.rectangle(annotated, (x, y), (x + w, y + h), SEARCH_COLOR, 2)
    cv2.putText(
        annotated, "enemy_search", (x, max(12, y - 6)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, SEARCH_COLOR, 1, cv2.LINE_AA,
    )
    return annotated


def measure_regions(frame: np.ndarray, config: WukongConfig) -> dict[str, float | bool | None]:
    """perceive 用到的原始测量结果（未做场景判定）。"""
    base = config.perception.base_resolution
    hud = config.hud
    detected = detect_bar(frame, hud.enemy_search, base)
    return {
        "hp_ratio": measure_bar(frame, hud.hp_bar, base),
        "hp_visible": match_ratio(frame, hud.hp_bar, base) >= config.combat.hp_visible_min,
        "stamina_ratio": measure_bar(frame, hud.stamina_bar, base),
        "mp_ratio": measure_bar(frame, hud.mp_bar, base),
        "mp_visible": match_ratio(frame, hud.mp_bar, base) >= config.combat.hp_visible_min,
        "boss_hp_ratio": measure_bar(frame, hud.enemy_hp_bar, base),
        "enemy_hp_dynamic": detected.ratio if detected is not None else None,
        "gourd_available": detect_presence(frame, hud.gourd, base),
        "dead_indicator": detect_presence(frame, hud.dead_indicator, base),
    }


def run_calibrate(
    config: WukongConfig,
    source: FrameSource,
    output_root: str | Path = "runs/",
) -> Path:
    """抓一帧并输出校准产物，返回输出目录。抓帧异常由调用方转成用户可读报错。"""
    out_dir = Path(output_root) / datetime.now().strftime("%Y%m%d-%H%M%S") / "calib"
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = source.grab()

    annotated_path = out_dir / "annotated.png"
    if not cv2.imwrite(str(annotated_path), annotate_frame(frame, config)):
        raise OSError(f"标注图写入失败: {annotated_path}")
    base = config.perception.base_resolution
    crop_specs = _region_items(config) + [("enemy_search", config.hud.enemy_search)]
    for name, spec in crop_specs:
        crop = crop_region(frame, spec.rect, base)
        if not cv2.imwrite(str(out_dir / f"{name}.png"), crop):
            raise OSError(f"区域裁剪写入失败: {out_dir / f'{name}.png'}")

    print("[calibrate] 测量值（基于 configs/wukong.yaml 当前 HUD 区域与阈值）:")
    for key, value in measure_regions(frame, config).items():
        if value is None:
            text = "-"
        elif isinstance(value, bool):
            text = "yes" if value else "no"
        else:
            text = f"{value:.2f}"
        print(f"  {key:<16} = {text}")
    print(f"[calibrate] 整图标注: {annotated_path}")
    print(f"[calibrate] 区域裁剪: {out_dir}/<区域名>.png")
    print("[calibrate] 若区域套错或测量值与画面不符，"
          "请修改 configs/wukong.yaml 的 hud.* 坐标/阈值后重跑 --calibrate")
    return out_dir
