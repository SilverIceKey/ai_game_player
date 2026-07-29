"""调参补丁：tuning_suggestions → runs/<ts>/tuning_suggestion.yaml。

红线（M2 计划第 5 节）：LLM 建议只写补丁文件，绝不自动改 configs/，
由人审阅后手动合入 configs/wukong.yaml。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml


def _scalar(value: object) -> str:
    """把建议值渲染成 YAML 标量文本（去掉 safe_dump 追加的文档结束符）。"""
    text = yaml.safe_dump(value, default_flow_style=True, allow_unicode=True).strip()
    if text.endswith("..."):
        text = text[:-3].strip()
    return text


def render_tuning_patch(
    suggestions: dict[str, object],
    issues: list[str] | None = None,
    reasons: dict[str, str] | None = None,
    source: str = "",
) -> str:
    """把 {"段.字段": 值} 展开为嵌套 YAML 文本，每条建议附理由注释。"""
    reasons = reasons or {}
    header = [
        "# M2 Ollama 复盘调参建议（补丁文件，不会自动应用）",
        f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        + (f"，来源: {source}" if source else ""),
        "# 请逐条核对后手动合入 configs/wukong.yaml",
    ]
    if issues:
        header.append("#")
        header.append("# 复盘发现的问题:")
        header.extend(f"#   - {issue}" for issue in issues)

    # 按段分组（校验键必须是 段.字段 两级，非法键降级为注释保留证据）
    sections: dict[str, list[tuple[str, object, str]]] = {}
    rejected: list[str] = []
    for key in sorted(suggestions):
        parts = str(key).split(".")
        if len(parts) != 2 or not all(parts):
            rejected.append(f"#   - 非法参数路径 {key!r}（建议值 {_scalar(suggestions[key])}），已跳过")
            continue
        reason = reasons.get(str(key), "")
        sections.setdefault(parts[0], []).append((parts[1], suggestions[key], reason))

    lines = header
    if rejected:
        lines.append("#")
        lines.append("# 以下建议参数路径不合法（应为 段.字段 两级），未写入补丁:")
        lines.extend(rejected)
    lines.append("")
    for section in sorted(sections):
        lines.append(f"{section}:")
        for field, value, reason in sections[section]:
            comment = f"  # 理由: {reason}" if reason else ""
            lines.append(f"  {field}: {_scalar(value)}{comment}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_tuning_patch(
    suggestions: dict[str, object],
    out_path: str | Path,
    issues: list[str] | None = None,
    reasons: dict[str, str] | None = None,
    source: str = "",
) -> Path:
    """写补丁文件，返回路径。suggestions 为空时不写文件（返回 None 由调用方判断）。"""
    if not suggestions:
        raise ValueError("无调参建议，不应生成补丁文件")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_tuning_patch(suggestions, issues, reasons, source), encoding="utf-8")
    return out
