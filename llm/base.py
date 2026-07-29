"""LLM 离线服务接口契约。

LLM 角色边界：只做复盘分析与参数调整建议，不进实时决策链路。
输入仅来自 core/recorder 的回放产物；输出经人确认后写入 configs/。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """统一 LLM 调用接口。

    默认实现为本地 Ollama；Kimi / OpenAI 通过 OpenAI 兼容协议接入。
    """

    def complete(self, prompt: str) -> str: ...


@dataclass
class ReviewReport:
    """复盘报告：诊断 + 参数调整建议。"""

    summary: str
    issues: list[str] = field(default_factory=list)
    tuning_suggestions: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class ReviewEngine(Protocol):
    """复盘引擎：输入回放文件，输出复盘报告。"""

    def review(self, replay: Path) -> ReviewReport: ...
