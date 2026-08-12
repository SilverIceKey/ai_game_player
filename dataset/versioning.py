"""Dataset 版本管理（spec §29）。

版本号固定格式 dataset-vNNN（至少 3 位数字，单调递增）。
每个数据集版本落盘一份元信息 JSON，训练侧必须记录 Model Version /
Dataset Version / Code Commit / Training Config / Evaluation Result，否则不可复现。
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_VERSION_RE = re.compile(r"^dataset-v(\d{3,})$")


def parse_dataset_version(version: str) -> int:
    """解析 dataset-vNNN，返回序号；格式非法抛 ValueError。"""
    match = _VERSION_RE.match(version)
    if match is None:
        raise ValueError(f"非法数据集版本号: {version!r}（格式应为 dataset-vNNN，如 dataset-v001）")
    return int(match.group(1))


def next_dataset_version(existing: list[str]) -> str:
    """在已有版本列表基础上生成下一个版本号；空列表返回 dataset-v001。"""
    if not existing:
        return "dataset-v001"
    return f"dataset-v{max(parse_dataset_version(v) for v in existing) + 1:03d}"


@dataclass
class DatasetVersionMeta:
    """数据集版本元信息（随数据集落盘为 JSON）。"""

    version: str
    created_us: int
    session_ids: list[str] = field(default_factory=list)
    sample_counts: dict[str, int] = field(default_factory=dict)  # 类别 -> 样本数
    description: str = ""

    def __post_init__(self) -> None:
        parse_dataset_version(self.version)  # 构造即校验版本号格式

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> DatasetVersionMeta:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"数据集版本元信息不存在: {p}")
        try:
            data: Any = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"数据集版本元信息损坏: {p} 第 {exc.lineno} 行: {exc.msg}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"数据集版本元信息必须是 JSON 对象: {p}")
        try:
            return cls(
                version=data["version"],
                created_us=int(data["created_us"]),
                session_ids=[str(s) for s in data.get("session_ids", [])],
                sample_counts={str(k): int(v) for k, v in data.get("sample_counts", {}).items()},
                description=str(data.get("description", "")),
            )
        except KeyError as exc:
            raise ValueError(f"数据集版本元信息缺少必填字段 {exc}: {p}") from exc
