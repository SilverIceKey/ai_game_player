"""黑神话悟空适配器：HUD 解析 + 场景感知 → GameState（计划文档 3.1 节）。

perceive = HUD 固定区域颜色检测（血条/体力/葫芦/敌方血条/死亡提示）
         + 光流里程计位姿 + 可通行区域评分。
区域坐标与阈值全部来自 configs/wukong.yaml（默认 1920x1080，首次实机校准）。
"""
from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.config import ConfigError, load_yaml_file, require
from core.contracts import Action, GameState
from core.control.directinput import ControlParams
from core.decision.navigation import ExplorationParams
from core.perception.bars import BarSearchSpec, detect_bar
from core.perception.odometry import OdometryParams, VisualOdometry
from core.perception.regions import (
    BarSpec,
    ColorRange,
    PresenceSpec,
    detect_presence,
    match_ratio,
    measure_bar,
)
from core.perception.walkable import WalkableAnalyzer, WalkableParams

_REQUIRED_KEYS = ("move_forward", "light_attack", "dodge", "heal", "lock_on")


_CAPTURE_BACKENDS = ("auto", "mss", "wgc")


@dataclass(frozen=True)
class WindowConfig:
    title: str
    rect: tuple[int, int, int, int] | None = None  # 手动指定截屏区域，跳过窗口定位
    capture_backend: str = "auto"  # 截屏后端：auto（WGC 优先降级 mss）/ mss / wgc
    foreground_on_start: bool = True  # 正式跑（非 dry-run/calibrate）启动时把游戏窗口提前台


@dataclass(frozen=True)
class HudConfig:
    hp_bar: BarSpec
    stamina_bar: BarSpec
    mp_bar: BarSpec  # 法力条（法术资源；与 HP 条同显同隐，只做感知/日志/回放，不接决策）
    enemy_hp_bar: BarSpec  # Boss 固定血条（位置固定）；与普通小怪动态血条互补，优先采用
    enemy_search: BarSearchSpec  # 普通小怪动态血条搜索区域（浮头血条，位置不固定）
    gourd: PresenceSpec
    dead_indicator: PresenceSpec


@dataclass(frozen=True)
class PerceptionConfig:
    base_resolution: tuple[int, int]
    odometry: OdometryParams
    walkable: WalkableParams


@dataclass(frozen=True)
class CombatParams:
    heal_hp_threshold: float = 0.35  # 血量低于该值且有葫芦 → 喝药
    engage_approach_ticks: int = 10  # 接战后接近 tick 数（近身判定占位，实机校准）
    dodge_interval_ticks: int = 12  # 战斗固定节奏闪避间隔
    dodge_on_hit_drop: float = 0.15  # 单 tick 血量跌幅达到该值判定为受击 → 闪避
    enemy_lost_ticks: int = 8  # 敌方血条连续消失 tick 数 → 脱战
    loot_wait_ticks: int = 20  # 掉落/脱战等待 tick 数
    enemy_present_min: float = 0.02  # 敌方血条填充超过该值视为接敌
    hp_visible_min: float = 0.02  # 自身血条区域匹配像素占比达到该值视为血条可见（非战斗隐藏）


@dataclass(frozen=True)
class DryRunParams:
    frame_interval_ticks: int = 50  # 干跑模式周期性抽样落帧间隔（tick）


@dataclass(frozen=True)
class WukongConfig:
    window: WindowConfig
    perception: PerceptionConfig
    hud: HudConfig
    combat: CombatParams
    keys: dict[str, str]
    control: ControlParams
    exploration: ExplorationParams
    dry_run: DryRunParams = DryRunParams()

    @classmethod
    def load(cls, path: str | Path) -> WukongConfig:
        return cls.from_dict(load_yaml_file(path))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WukongConfig:
        window = require(data, "window", "")
        rect = window.get("rect") if isinstance(window, dict) else None
        backend = window.get("capture_backend", "auto") if isinstance(window, dict) else "auto"
        if backend not in _CAPTURE_BACKENDS:
            raise ConfigError(
                f"window.capture_backend 非法: {backend!r}（仅支持 {'/'.join(_CAPTURE_BACKENDS)}）"
            )
        window_cfg = WindowConfig(
            title=str(require(window, "title", "window")),
            rect=_rect4(rect, "window.rect") if rect is not None else None,
            capture_backend=str(backend),
            foreground_on_start=(
                bool(window.get("foreground_on_start", True)) if isinstance(window, dict) else True
            ),
        )

        perception = data.get("perception") or {}
        base_res = perception.get("base_resolution", [1920, 1080])
        if not isinstance(base_res, (list, tuple)) or len(base_res) != 2:
            raise ConfigError("perception.base_resolution 必须是 [宽, 高] 二元列表")
        perception_cfg = PerceptionConfig(
            base_resolution=(int(base_res[0]), int(base_res[1])),
            odometry=_params(OdometryParams, perception.get("odometry"), "perception.odometry"),
            walkable=_params(WalkableParams, perception.get("walkable"), "perception.walkable"),
        )

        hud = require(data, "hud", "")
        hud_cfg = HudConfig(
            hp_bar=_bar(require(hud, "hp_bar", "hud"), "hud.hp_bar"),
            stamina_bar=_bar(require(hud, "stamina_bar", "hud"), "hud.stamina_bar"),
            mp_bar=_bar(require(hud, "mp_bar", "hud"), "hud.mp_bar"),
            enemy_hp_bar=_bar(require(hud, "enemy_hp_bar", "hud"), "hud.enemy_hp_bar"),
            enemy_search=_search(require(hud, "enemy_search", "hud"), "hud.enemy_search"),
            gourd=_presence(require(hud, "gourd", "hud"), "hud.gourd"),
            dead_indicator=_presence(require(hud, "dead_indicator", "hud"), "hud.dead_indicator"),
        )

        keys = require(data, "keys", "")
        missing = [k for k in _REQUIRED_KEYS if not (isinstance(keys, dict) and keys.get(k))]
        if missing:
            raise ConfigError(f"配置缺少必填字段: keys.{missing[0]}")

        return cls(
            window=window_cfg,
            perception=perception_cfg,
            hud=hud_cfg,
            combat=_params(CombatParams, data.get("combat"), "combat"),
            keys={str(k): str(v) for k, v in keys.items()},
            control=_params(ControlParams, data.get("control"), "control"),
            exploration=_params(ExplorationParams, data.get("exploration"), "exploration"),
            dry_run=_params(DryRunParams, data.get("dry_run"), "dry_run"),
        )


def _params(cls, raw: Any, ctx: str):
    """把 YAML 子映射填进 dataclass（缺省字段用默认值，未知字段报错防止配置笔误）。"""
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{ctx} 必须是映射")
    fields = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(raw) - fields)
    if unknown:
        raise ConfigError(f"{ctx} 存在未知字段: {', '.join(unknown)}")
    return cls(**raw)


def _rect4(raw: Any, ctx: str) -> tuple[int, int, int, int]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ConfigError(f"{ctx} 必须是 [x, y, w, h] 四元列表")
    return tuple(int(v) for v in raw)  # type: ignore[return-value]


def _color(raw: Any, ctx: str) -> ColorRange:
    lower = require(raw, "hsv_lower", ctx)
    upper = require(raw, "hsv_upper", ctx)
    if len(lower) != 3 or len(upper) != 3:
        raise ConfigError(f"{ctx} 的 HSV 阈值必须是 3 元素列表")
    return ColorRange(
        tuple(int(v) for v in lower),  # type: ignore[arg-type]
        tuple(int(v) for v in upper),  # type: ignore[arg-type]
    )


def _bar(raw: Any, ctx: str) -> BarSpec:
    return BarSpec(
        rect=_rect4(require(raw, "rect", ctx), f"{ctx}.rect"),
        color=_color(raw, ctx),
        column_fill=float(raw.get("column_fill", 0.3)),
    )


def _presence(raw: Any, ctx: str) -> PresenceSpec:
    return PresenceSpec(
        rect=_rect4(require(raw, "rect", ctx), f"{ctx}.rect"),
        color=_color(raw, ctx),
        min_ratio=float(raw.get("presence_ratio", 0.05)),
    )


def _search(raw: Any, ctx: str) -> BarSearchSpec:
    track = None
    if raw.get("track_hsv_lower") is not None and raw.get("track_hsv_upper") is not None:
        lower, upper = raw["track_hsv_lower"], raw["track_hsv_upper"]
        if len(lower) != 3 or len(upper) != 3:
            raise ConfigError(f"{ctx} 的 track HSV 阈值必须是 3 元素列表")
        track = ColorRange(tuple(int(v) for v in lower), tuple(int(v) for v in upper))  # type: ignore[arg-type]
    return BarSearchSpec(
        rect=_rect4(require(raw, "rect", ctx), f"{ctx}.rect"),
        color=_color(raw, ctx),
        track_color=track,
        min_length=int(raw.get("min_length", 40)),
        min_aspect=float(raw.get("min_aspect", 3.0)),
        min_fill=float(raw.get("min_fill", 0.5)),
        column_fill=float(raw.get("column_fill", 0.3)),
    )


class WukongAdapter:
    """实现 games.base.GameAdapter 契约。"""

    ACTION_SPACE = ("move", "turn", "light_attack", "dodge", "heal", "lock_on")

    def __init__(self, config: WukongConfig):
        self.config = config
        self.odometry = VisualOdometry(config.perception.odometry)
        self.walkable = WalkableAnalyzer(config.perception.walkable)

    def perceive(self, frame: np.ndarray) -> GameState:
        cfg = self.config
        base = cfg.perception.base_resolution

        # 自身血条非战斗隐藏（计划 3.1a）：不可见时 hp 按 1.0 处理（无读数即无伤）
        hp_visible = match_ratio(frame, cfg.hud.hp_bar, base) >= cfg.combat.hp_visible_min
        hp = measure_bar(frame, cfg.hud.hp_bar, base) if hp_visible else 1.0
        stamina = measure_bar(frame, cfg.hud.stamina_bar, base)
        # 法力条与 HP 条同显同隐：只做感知/日志/回放（M1 决策不放法术）
        mp_visible = match_ratio(frame, cfg.hud.mp_bar, base) >= cfg.combat.hp_visible_min
        mp = measure_bar(frame, cfg.hud.mp_bar, base) if mp_visible else 1.0

        # 敌方血条：Boss 固定条优先，其次普通小怪动态浮头血条（区域检测）
        boss_hp = measure_bar(frame, cfg.hud.enemy_hp_bar, base)
        boss_present = boss_hp >= cfg.combat.enemy_present_min
        detected = detect_bar(frame, cfg.hud.enemy_search, base)
        if boss_present:
            enemy_hp: float | None = boss_hp
            enemy_present = True
            enemy_source: str | None = "boss"
            enemy_box = cfg.hud.enemy_hp_bar.rect
        elif detected is not None:
            enemy_hp = detected.ratio
            enemy_present = True
            enemy_source = "dynamic"
            enemy_box = detected.box
        else:
            enemy_hp = None
            enemy_present = False
            enemy_source = None
            enemy_box = None

        gourd = detect_presence(frame, cfg.hud.gourd, base)
        dead = detect_presence(frame, cfg.hud.dead_indicator, base)

        pose = self.odometry.update(frame)
        walk = self.walkable.analyze(frame)

        if dead:
            scene = "dead"
        elif enemy_present:
            scene = "combat"
        else:
            scene = "explore"

        return GameState(
            timestamp=time.time(),
            scene=scene,
            raw={
                "hp_ratio": hp,
                "hp_visible": hp_visible,
                "stamina_ratio": stamina,
                "mp_ratio": mp,
                "mp_visible": mp_visible,
                "gourd_available": gourd,
                "enemy_hp_ratio": enemy_hp,
                "enemy_hp_source": enemy_source,
                "enemy_bar_box": enemy_box,
                "enemy_present": enemy_present,
                "in_combat": enemy_present,
                "pose": pose.as_tuple(),
                "walkable": walk.as_dict(),
            },
        )

    def available_actions(self, state: GameState) -> list[Action]:
        if state.scene == "dead":
            return []
        if state.scene == "combat":
            names = ("light_attack", "dodge", "heal", "lock_on", "move")
        else:
            names = ("move", "turn", "lock_on")
        return [Action(name) for name in names]

    def action_space(self) -> list[str]:
        return list(self.ACTION_SPACE)
