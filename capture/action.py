"""NormalizedAction 契约（spec §9：内部动作表示必须与实际键位解耦）。

全项目共享：capture（采集）/ dataset（存储）/ model（输出）/ runtime（执行）。
动作空间固定为 4 个连续轴 + 布尔按钮集；键位映射由 config 的 keys 段与
runtime 的 InputAdapter 负责，模型与数据集不认识任何具体键位语义。

动作来源（spec §26/§27）：
- human：OBSERVE_TRAIN 中玩家示范
- correction：AUTOPILOT 人工接管后的纠正操作（DAgger 闭环核心数据）
- ai：AUTOPILOT 中模型输出（SHADOW / 闭环评估用）
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any

# spec §9 动作空间的全部布尔按钮（顺序即二进制掩码位序，禁止随意改动）
BUTTONS: tuple[str, ...] = (
    "attack_light",
    "attack_heavy",
    "dodge",
    "block",
    "parry",
    "jump",
    "interact",
    "heal",
    "skill_1",
    "skill_2",
    "skill_3",
    "skill_4",
    "lock_target",
    "wait",
)
_BUTTON_INDEX = {name: i for i, name in enumerate(BUTTONS)}

SOURCE_HUMAN = "human"
SOURCE_CORRECTION = "correction"
SOURCE_AI = "ai"
SOURCES: tuple[str, ...] = (SOURCE_HUMAN, SOURCE_CORRECTION, SOURCE_AI)
_SOURCE_INDEX = {name: i for i, name in enumerate(SOURCES)}

# actions.bin 定长记录（spec §20.3：高频 Action Record 用 binary format）：
# timestamp_us(Q) + 4 轴(4f) + 按钮掩码(H) + 来源(B) = 27 字节
ACTION_BIN = struct.Struct("<Q4fHB")


def _clamp(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


@dataclass(frozen=True)
class NormalizedAction:
    """单步归一化动作。连续轴范围 [-1, 1]，越界自动截断。"""

    move_x: float = 0.0  # 右为正
    move_y: float = 0.0  # 前为正
    camera_x: float = 0.0  # 右转视角为正
    camera_y: float = 0.0  # 上抬视角为正
    buttons: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "move_x", _clamp(self.move_x))
        object.__setattr__(self, "move_y", _clamp(self.move_y))
        object.__setattr__(self, "camera_x", _clamp(self.camera_x))
        object.__setattr__(self, "camera_y", _clamp(self.camera_y))
        buttons = frozenset(self.buttons)
        unknown = buttons - _BUTTON_INDEX.keys()
        if unknown:
            raise ValueError(f"未知动作按钮: {sorted(unknown)}（合法值见 capture.action.BUTTONS）")
        object.__setattr__(self, "buttons", buttons)

    @classmethod
    def neutral(cls) -> NormalizedAction:
        return cls()

    def pressed(self, name: str) -> bool:
        return name in self.buttons

    def is_neutral(self) -> bool:
        return (
            not self.buttons
            and self.move_x == 0.0
            and self.move_y == 0.0
            and self.camera_x == 0.0
            and self.camera_y == 0.0
        )

    @property
    def button_mask(self) -> int:
        mask = 0
        for name in self.buttons:
            mask |= 1 << _BUTTON_INDEX[name]
        return mask

    @classmethod
    def from_mask(
        cls,
        mask: int,
        move_x: float = 0.0,
        move_y: float = 0.0,
        camera_x: float = 0.0,
        camera_y: float = 0.0,
    ) -> NormalizedAction:
        buttons = frozenset(name for name, i in _BUTTON_INDEX.items() if mask & (1 << i))
        return cls(move_x, move_y, camera_x, camera_y, buttons)

    def to_dict(self) -> dict[str, Any]:
        """调试/导出用 JSON 表示（spec §20.3：JSON 仅用于调试和导出）。"""
        return {
            "move_x": self.move_x,
            "move_y": self.move_y,
            "camera_x": self.camera_x,
            "camera_y": self.camera_y,
            **{name: name in self.buttons for name in BUTTONS},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NormalizedAction:
        buttons = frozenset(name for name in BUTTONS if data.get(name))
        return cls(
            move_x=float(data.get("move_x", 0.0)),
            move_y=float(data.get("move_y", 0.0)),
            camera_x=float(data.get("camera_x", 0.0)),
            camera_y=float(data.get("camera_y", 0.0)),
            buttons=buttons,
        )


@dataclass(frozen=True)
class ActionRecord:
    """带时间戳与来源的动作记录（采集与存储的基本单元）。"""

    timestamp_us: int
    action: NormalizedAction
    source: str = SOURCE_HUMAN

    RECORD_SIZE = ACTION_BIN.size

    def __post_init__(self) -> None:
        if self.source not in _SOURCE_INDEX:
            raise ValueError(f"未知动作来源: {self.source!r}（合法值: {SOURCES}）")

    def pack(self) -> bytes:
        a = self.action
        return ACTION_BIN.pack(
            self.timestamp_us, a.move_x, a.move_y, a.camera_x, a.camera_y,
            a.button_mask, _SOURCE_INDEX[self.source],
        )

    @classmethod
    def unpack(cls, buf: bytes) -> ActionRecord:
        ts, mx, my, cx, cy, mask, src = ACTION_BIN.unpack(buf)
        return cls(
            timestamp_us=ts,
            action=NormalizedAction.from_mask(mask, mx, my, cx, cy),
            source=SOURCES[src],
        )


@dataclass(frozen=True)
class ActionChunk:
    """一次推理输出的未来动作序列（spec §15 Action Chunking）。"""

    actions: tuple[NormalizedAction, ...]
    step_ms: float = 50.0
    model_version: str = ""
    confidence: dict[str, float] = field(default_factory=dict)
    created_us: int = 0

    def __post_init__(self) -> None:
        if not self.actions:
            raise ValueError("ActionChunk 至少包含一步动作")
        object.__setattr__(self, "actions", tuple(self.actions))
