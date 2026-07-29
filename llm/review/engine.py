"""复盘引擎：采样帧 ↔ 操作窗口配对 → 分批送 Ollama → 汇总 ReviewReport。

输入 run_dir（frames/*.png + replay.jsonl + session.log）：
- 帧文件名 {tick:06d}_{tag}.png 解析出 tick，按 review.window_ticks 从回放取操作窗口
- 按 review.batch_size 分批送视觉模型（帧图 + prompt）
- 每批要求输出固定 JSON；解析失败的批降级为纯文本摘要
- 汇总为 llm.base.ReviewReport（tuning_suggestions 供 llm/tuning 写补丁）
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from core.config import ReviewConfig
from core.recorder.jsonl import read_replay
from llm.base import LLMProvider, ReviewReport
from llm.review.prompts import build_batch_prompt

_FRAME_RE = re.compile(r"^(\d{6})_(.+)\.png$")
# 兼容旧命名：早期版本回放文件叫 session.jsonl
_REPLAY_CANDIDATES = ("replay.jsonl", "session.jsonl")


@dataclass(frozen=True)
class SampledFrame:
    path: Path
    tick: int
    tag: str


def list_sampled_frames(run_dir: str | Path) -> list[SampledFrame]:
    """扫描 frames/ 下符合 {tick:06d}_{tag}.png 约定的帧，按 tick 排序。"""
    frames_dir = Path(run_dir) / "frames"
    if not frames_dir.is_dir():
        return []
    frames = []
    for path in frames_dir.iterdir():
        m = _FRAME_RE.match(path.name)
        if m:
            frames.append(SampledFrame(path=path, tick=int(m.group(1)), tag=m.group(2)))
    return sorted(frames, key=lambda f: f.tick)


def parse_json_payload(text: str) -> dict | None:
    """从模型输出提取 JSON 对象：整串解析 → ```json 代码块 → 最外层花括号子串。"""
    text = text.strip()
    if not text:
        return None
    candidates = [text]
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})```", text, flags=re.DOTALL)
    candidates.extend(fenced)
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def find_replay_file(run_dir: str | Path) -> Path:
    """定位 run 目录下的回放 JSONL（新命名 replay.jsonl 优先，兼容 session.jsonl）。"""
    run_dir = Path(run_dir)
    for name in _REPLAY_CANDIDATES:
        candidate = run_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"回放文件不存在: {run_dir} 下未找到 {' / '.join(_REPLAY_CANDIDATES)}"
    )


class OllamaReviewEngine:
    """实现 llm.base.ReviewEngine 契约。review(replay) 的 replay 为 run 目录路径。"""

    def __init__(
        self,
        provider: LLMProvider,
        review_config: ReviewConfig | None = None,
        game_config_path: str | Path | None = None,
    ):
        self.provider = provider
        self.config = review_config or ReviewConfig()
        self.game_config_path = Path(game_config_path) if game_config_path else None
        # 最近一次 review 的调参理由（键同 tuning_suggestions），供补丁注释用
        self.suggestion_reasons: dict[str, str] = {}

    def review(self, replay: Path) -> ReviewReport:
        run_dir = Path(replay)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"复盘目录不存在: {run_dir}")
        frames = list_sampled_frames(run_dir)
        if not frames:
            return ReviewReport(summary=f"无可复盘样本: {run_dir}/frames/ 为空（先跑一段采样）")

        records = read_replay(find_replay_file(run_dir))
        config_excerpt = self._load_config_excerpt()

        self.suggestion_reasons = {}
        summaries: list[str] = []
        issues: list[str] = []
        suggestions: dict[str, object] = {}
        batches = [
            frames[i : i + self.config.batch_size]
            for i in range(0, len(frames), self.config.batch_size)
        ]
        for index, batch in enumerate(batches, start=1):
            prompt = build_batch_prompt(batch, records, self.config.window_ticks, config_excerpt)
            raw = self.provider.complete_with_images(  # type: ignore[attr-defined]
                prompt, [f.path for f in batch]
            )
            payload = parse_json_payload(raw)
            if payload is None:
                # 降级：纯文本摘要，不丢信息
                summaries.append(f"[第 {index} 批·文本摘要] {raw.strip()}")
                continue
            summary = str(payload.get("summary", "")).strip()
            if summary:
                summaries.append(f"[第 {index} 批] {summary}")
            issues.extend(str(item) for item in payload.get("issues") or [])
            for key, value in (payload.get("tuning_suggestions") or {}).items():
                suggestions[str(key)] = value  # 后批覆盖先批（模型迭代意见以新为准）
            for key, reason in (payload.get("suggestion_reasons") or {}).items():
                self.suggestion_reasons[str(key)] = str(reason)

        return ReviewReport(
            summary="\n".join(summaries) if summaries else "（模型未产出有效摘要）",
            issues=issues,
            tuning_suggestions=suggestions,
        )

    def _load_config_excerpt(self) -> str:
        if self.game_config_path and self.game_config_path.is_file():
            return self.game_config_path.read_text(encoding="utf-8")
        return "（未提供当前配置，tuning_suggestions 请基于日志推断参数路径）"
