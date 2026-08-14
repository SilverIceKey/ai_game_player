"""模型版本元数据（spec §29：每个模型必须记录五项，否则不可复现）。

本文件只负责元数据的结构化读写（JSON）；candidate → promote 的
状态流转与 active 路径管理由 train/registry.py 负责。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from capture.clock import now_us


@dataclass(frozen=True)
class ModelCheckpointMeta:
    """模型 checkpoint 元数据（spec §29）。"""

    model_version: str
    dataset_version: str
    code_commit: str
    training_config: dict[str, Any] = field(default_factory=dict)
    eval_result: dict[str, Any] = field(default_factory=dict)
    created_us: int = 0
    epoch: int | None = None
    train_loss: dict[str, float] = field(default_factory=dict)
    total_loss: float | None = None
    gate: dict[str, float] = field(default_factory=dict)
    available_epoch_checkpoints: tuple[str, ...] = ()
    selected_epoch: int | None = None
    selection_reason: str = ""

    def __post_init__(self) -> None:
        if not self.model_version.strip():
            raise ValueError("model_version 不能为空")
        if not self.dataset_version.strip():
            raise ValueError("dataset_version 不能为空（spec §29：dataset-vNNN）")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "dataset_version": self.dataset_version,
            "code_commit": self.code_commit,
            "training_config": self.training_config,
            "eval_result": self.eval_result,
            "created_us": self.created_us,
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "total_loss": self.total_loss,
            "gate": self.gate,
            "available_epoch_checkpoints": list(self.available_epoch_checkpoints),
            "selected_epoch": self.selected_epoch,
            "selection_reason": self.selection_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelCheckpointMeta:
        if not isinstance(data, dict):
            raise ValueError(f"checkpoint 元数据必须是 JSON 对象，实际为 {type(data).__name__}")
        try:
            return cls(
                model_version=str(data["model_version"]),
                dataset_version=str(data["dataset_version"]),
                code_commit=str(data.get("code_commit", "")),
                training_config=dict(data.get("training_config") or {}),
                eval_result=dict(data.get("eval_result") or {}),
                created_us=int(data.get("created_us", 0)),
                epoch=(int(data["epoch"]) if data.get("epoch") is not None else None),
                train_loss={
                    str(k): float(v) for k, v in dict(data.get("train_loss") or {}).items()
                },
                total_loss=(
                    float(data["total_loss"])
                    if data.get("total_loss") is not None
                    else None
                ),
                gate={str(k): float(v) for k, v in dict(data.get("gate") or {}).items()},
                available_epoch_checkpoints=tuple(
                    str(v) for v in data.get("available_epoch_checkpoints") or ()
                ),
                selected_epoch=(
                    int(data["selected_epoch"])
                    if data.get("selected_epoch") is not None
                    else None
                ),
                selection_reason=str(data.get("selection_reason", "")),
            )
        except KeyError as exc:
            raise ValueError(f"checkpoint 元数据缺少必填字段: {exc}") from exc

    def save(self, path: str | Path) -> Path:
        """写 JSON 元数据文件（UTF-8，缩进可读）。"""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return out

    @classmethod
    def load(cls, path: str | Path) -> ModelCheckpointMeta:
        """读 JSON 元数据；文件损坏给明确报错，不静默兜底。"""
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"checkpoint 元数据文件不存在: {p}")
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"checkpoint 元数据 JSON 损坏: {p}: {exc}") from exc
        try:
            return cls.from_dict(data)
        except ValueError as exc:
            raise ValueError(f"checkpoint 元数据非法: {p}: {exc}") from exc


def new_checkpoint_meta(
    model_version: str,
    dataset_version: str,
    code_commit: str,
    training_config: dict[str, Any] | None = None,
) -> ModelCheckpointMeta:
    """创建新 checkpoint 元数据（created_us 打统一时钟）。"""
    return ModelCheckpointMeta(
        model_version=model_version,
        dataset_version=dataset_version,
        code_commit=code_commit,
        training_config=training_config or {},
        eval_result={},
        created_us=now_us(),
    )
