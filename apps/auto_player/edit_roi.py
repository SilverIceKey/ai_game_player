"""--edit-roi 交互式 ROI 校准（计划 3.1a 节：替代手填坐标）。

工作流：抓一帧（与 calibrate 同链路）→ 控制台菜单选择要配的区域 →
cv2.selectROI 在帧上鼠标拖框 → 坐标反缩放回基准分辨率 → q 退出时统一写回
configs/wukong.yaml（ruamel.yaml 保留注释与格式）。

GUI 交互（selectROI）仅 Windows 实机可用，保持薄；
坐标换算、菜单解析、配置写回均为可测纯逻辑。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.perception.base import FrameSource
from core.perception.regions import scale_rect

# (配置键, 菜单标签)；键即 hud 段下的字段名
EDITABLE_ITEMS: list[tuple[str, str]] = [
    ("hp_bar", "自身血条 hp_bar"),
    ("stamina_bar", "体力条 stamina_bar"),
    ("gourd", "葫芦图标 gourd"),
    ("dead_indicator", "死亡提示 dead_indicator"),
    ("enemy_hp_bar", "Boss 固定血条 enemy_hp_bar"),
    ("enemy_search", "小怪血条搜索区域 enemy_search"),
]

_ITEM_COLORS = [
    (0, 255, 255), (0, 200, 255), (255, 0, 255),
    (255, 0, 128), (0, 128, 255), (255, 255, 0),
]


def parse_menu_choice(text: str, count: int) -> int | None:
    """菜单输入解析：'q' → None（保存退出）；'1'..count → 0-based 索引；其余 ValueError。"""
    text = text.strip().lower()
    if text in ("q", "quit", "exit"):
        return None
    try:
        n = int(text)
    except ValueError:
        raise ValueError(f"无效输入: {text!r}（输入 1-{count} 选择区域，或 q 保存退出）")
    if not 1 <= n <= count:
        raise ValueError(f"编号超出范围: {n}（1-{count} 或 q）")
    return n - 1


def rect_to_base(
    rect: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
    base_resolution: tuple[int, int] = (1920, 1080),
) -> tuple[int, int, int, int]:
    """帧坐标 rect → 基准分辨率坐标（regions.scale_rect 的互逆换算），并裁剪到基准范围内。"""
    h, w = frame_shape[:2]
    bw, bh = base_resolution
    sx, sy = w / bw, h / bh
    x, y, rw, rh = rect
    bx = min(bw - 1, max(0, int(round(x / sx))))
    by = min(bh - 1, max(0, int(round(y / sy))))
    brw = min(bw - bx, max(1, int(round(rw / sx))))
    brh = min(bh - by, max(1, int(round(rh / sy))))
    return bx, by, brw, brh


def update_yaml_rect(yaml_path: str | Path, key: str, rect: tuple[int, int, int, int]) -> None:
    """把 hud.<key>.rect 写回 YAML（ruamel 往返，保留注释与格式）。"""
    from ruamel.yaml import YAML

    path = Path(yaml_path)
    yaml = YAML()
    yaml.preserve_quotes = True
    doc = yaml.load(path.read_text(encoding="utf-8"))
    if key not in (doc.get("hud") or {}):
        raise KeyError(f"配置中不存在 hud.{key}（可配项: {', '.join(doc.get('hud') or {})}）")
    doc["hud"][key]["rect"] = [int(v) for v in rect]
    with path.open("w", encoding="utf-8") as fp:
        yaml.dump(doc, fp)


def _draw_reference(
    frame: np.ndarray,
    doc,
    base_resolution: tuple[int, int],
    highlight: str,
) -> np.ndarray:
    """在帧上画出所有已配置区域的当前矩形作为拖框参考，当前配置项高亮。"""
    annotated = frame.copy()
    hud = doc.get("hud") or {}
    for i, (key, label) in enumerate(EDITABLE_ITEMS):
        rect = (hud.get(key) or {}).get("rect")
        if not rect or len(rect) != 4:
            continue
        x, y, w, h = scale_rect(tuple(int(v) for v in rect), frame.shape, base_resolution)
        color = _ITEM_COLORS[i % len(_ITEM_COLORS)]
        thickness = 3 if key == highlight else 1
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(annotated, label, (x, max(12, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return annotated


def run_edit_roi(
    config_path: str | Path,
    source: FrameSource,
    base_resolution: tuple[int, int] = (1920, 1080),
) -> int:
    """交互式校准主流程。返回退出码；GUI/抓帧异常由调用方转成用户可读报错。"""
    from ruamel.yaml import YAML

    config_path = Path(config_path)
    yaml = YAML()
    yaml.preserve_quotes = True
    doc = yaml.load(config_path.read_text(encoding="utf-8"))

    frame = source.grab()
    print("[edit-roi] 已抓帧。选择要校准的区域，在弹出窗口上鼠标拖框，回车确认，ESC 取消。")

    pending: dict[str, tuple[int, int, int, int]] = {}
    while True:
        print("\n可配置区域:")
        for i, (_, label) in enumerate(EDITABLE_ITEMS, start=1):
            mark = "（已修改未保存）" if EDITABLE_ITEMS[i - 1][0] in pending else ""
            print(f"  {i}. {label}{mark}")
        try:
            choice = parse_menu_choice(input("输入编号选择，q 保存并退出: "), len(EDITABLE_ITEMS))
        except ValueError as exc:
            print(f"[edit-roi] {exc}")
            continue
        if choice is None:
            break

        key, label = EDITABLE_ITEMS[choice]
        reference = _draw_reference(frame, doc, base_resolution, highlight=key)
        try:
            roi = cv2.selectROI(
                f"edit-roi: {label}（拖框后回车确认，ESC/c 取消）", reference, showCrosshair=True
            )
        except cv2.error as exc:
            raise RuntimeError(
                "cv2.selectROI 需要 GUI 版 opencv：Windows 实机请安装 opencv-python"
                f"（pyproject 已按平台分流）: {exc}"
            ) from exc
        finally:
            cv2.destroyAllWindows()
        x, y, w, h = (int(v) for v in roi)
        if w <= 0 or h <= 0:
            print(f"[edit-roi] 已取消，{label} 未修改")
            continue
        base_rect = rect_to_base((x, y, w, h), frame.shape, base_resolution)
        doc["hud"][key]["rect"] = list(base_rect)
        pending[key] = base_rect
        print(f"[edit-roi] {label} → rect={list(base_rect)}（q 退出时统一写回）")

    if pending:
        with config_path.open("w", encoding="utf-8") as fp:
            yaml.dump(doc, fp)
        print(f"[edit-roi] 已写回 {len(pending)} 项到 {config_path}")
    else:
        print("[edit-roi] 无修改，配置未动")
    return 0
