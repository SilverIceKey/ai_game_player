"""模型注册表（spec §7 模型生命周期：Candidate → Evaluate → Promote/Reject）。

状态机：

    candidate → evaluated → active（promote）
                    ↘ rejected（reject）

约束：
- 禁止训练后直接覆盖 active 模型：promote 必须显式调用且模型已评估；
- promote 只允许在 Episode Boundary 发生（spec §7：不能战斗/控制中热切）。
  本类不感知 episode 状态，**episode 边界由调用方保证**（docstring 约定）；
- promote 新模型时，原 active 模型降为 evaluated（保留可追溯）。

状态持久化为 registry.json；文件损坏给明确报错。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from model.checkpoint import ModelCheckpointMeta

# 模型状态（spec §7）
STATUS_CANDIDATE = "candidate"
STATUS_EVALUATED = "evaluated"
STATUS_ACTIVE = "active"
STATUS_REJECTED = "rejected"


class ModelRegistry:
    """候选/现役模型注册表，状态持久化到 registry.json。"""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        # model_version -> {"meta": ModelCheckpointMeta, "status": str}
        self._models: dict[str, dict[str, Any]] = {}
        self._active_version: str | None = None

    # ---------- 查询 ----------

    @property
    def path(self) -> Path:
        return self._path

    def status(self, model_version: str) -> str:
        return self._entry(model_version)["status"]

    def active_model(self) -> ModelCheckpointMeta | None:
        """当前 Active Model；无则 None。"""
        if self._active_version is None:
            return None
        return self._models[self._active_version]["meta"]

    def list_models(self) -> dict[str, str]:
        """{model_version: status} 快照。"""
        return {v: e["status"] for v, e in self._models.items()}

    # ---------- 状态流转 ----------

    def register_candidate(self, meta: ModelCheckpointMeta) -> None:
        """注册候选模型（spec §7：训练产物先成为 Candidate，不得直接覆盖 active）。"""
        if meta.model_version in self._models:
            raise ValueError(f"模型版本已注册: {meta.model_version}")
        self._models[meta.model_version] = {"meta": meta, "status": STATUS_CANDIDATE}
        self._save()

    def record_evaluation(self, model_version: str, result: dict[str, Any]) -> None:
        """记录离线/闭环评估结果（candidate/evaluated → evaluated）。"""
        entry = self._entry(model_version)
        if entry["status"] in (STATUS_ACTIVE, STATUS_REJECTED):
            raise ValueError(
                f"模型 {model_version} 状态为 {entry['status']}，不能再记录评估结果"
            )
        meta = entry["meta"]
        entry["meta"] = ModelCheckpointMeta(
            model_version=meta.model_version,
            dataset_version=meta.dataset_version,
            code_commit=meta.code_commit,
            training_config=meta.training_config,
            eval_result=dict(result),
            created_us=meta.created_us,
        )
        entry["status"] = STATUS_EVALUATED
        self._save()

    def promote(self, model_version: str) -> None:
        """提升候选为 Active Model。

        前置：模型状态为 evaluated（spec §7：必须先通过评估）。
        调用方保证只在 Episode Boundary 调用——不能在战斗/控制过程中热切。
        """
        entry = self._entry(model_version)
        if entry["status"] != STATUS_EVALUATED:
            raise ValueError(
                f"只有 evaluated 状态可 promote，模型 {model_version} 当前为 {entry['status']}"
            )
        if self._active_version is not None:
            self._models[self._active_version]["status"] = STATUS_EVALUATED
        entry["status"] = STATUS_ACTIVE
        self._active_version = model_version
        self._save()

    def reject(self, model_version: str) -> None:
        """拒绝候选（评估未通过，spec §7 Reject 分支）。"""
        entry = self._entry(model_version)
        if entry["status"] == STATUS_ACTIVE:
            raise ValueError(f"模型 {model_version} 是 active，不能 reject")
        entry["status"] = STATUS_REJECTED
        self._save()

    # ---------- 持久化 ----------

    @classmethod
    def load(cls, path: str | Path) -> ModelRegistry:
        """加载 registry.json；不存在则返回空注册表；损坏给明确报错。"""
        registry = cls(path)
        p = Path(path)
        if not p.exists():
            return registry
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"registry 文件 JSON 损坏: {p}: {exc}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("models"), dict):
            raise ValueError(f"registry 文件结构非法: {p}")
        for version, item in data["models"].items():
            meta = ModelCheckpointMeta.from_dict(item["meta"])
            status = item["status"]
            if status not in (STATUS_CANDIDATE, STATUS_EVALUATED, STATUS_ACTIVE, STATUS_REJECTED):
                raise ValueError(f"registry 文件含未知状态 {status!r}: {p}")
            registry._models[version] = {"meta": meta, "status": status}
        active = data.get("active_version")
        if active is not None:
            if active not in registry._models:
                raise ValueError(f"registry active_version 未注册: {active!r}: {p}")
            registry._active_version = active
        return registry

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "active_version": self._active_version,
            "models": {
                v: {"meta": e["meta"].to_dict(), "status": e["status"]}
                for v, e in self._models.items()
            },
        }
        self._path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def _entry(self, model_version: str) -> dict[str, Any]:
        entry = self._models.get(model_version)
        if entry is None:
            raise KeyError(f"模型未注册: {model_version}")
        return entry
