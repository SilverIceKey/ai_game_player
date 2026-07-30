"""复盘 prompt 模板：采样帧 + 操作窗口 + 当前参数 → 固定 JSON 诊断。

模型被要求只输出固定 JSON（summary / issues / tuning_suggestions / suggestion_reasons），
解析失败时引擎降级为纯文本摘要（见 llm/review/engine.py）。
"""
from __future__ import annotations

from core.recorder.jsonl import tick_window

# 输出 JSON 的固定结构说明（拼进 prompt，引擎按同一结构解析）
JSON_OUTPUT_SPEC = """\
{
  "summary": "本批帧的总体评价（中文，两三句）",
  "issues": ["发现的问题1", "发现的问题2"],
  "tuning_suggestions": {"段.字段": 建议的新值},
  "suggestion_reasons": {"段.字段": "该建议的理由"}
}
tuning_suggestions 的键必须取自上面 YAML 中出现的参数路径（如 combat.heal_hp_threshold、
exploration.turn_degrees），没有建议时输出空对象 {}；suggestion_reasons 与之一一对应。"""


def format_record_line(tick: int, record: dict) -> str:
    """把一条 JSONL 回放记录压缩成一行操作日志（供 prompt 与排查阅读）。"""
    state = record.get("state") or {}
    raw = state.get("raw") or {}
    output = record.get("output") or {}
    # suggestion 记录的 action 嵌套一层；action 记录字段平铺
    action_obj = (output.get("action") or {}) if output.get("type") == "suggestion" else output
    intent = (record.get("extra") or {}).get("intent", "")
    enemy = raw.get("enemy_hp_ratio")
    enemy_s = f"{enemy:.2f}" if isinstance(enemy, (int, float)) else "-"
    hp = raw.get("hp_ratio")
    hp_s = f"{hp:.2f}" if isinstance(hp, (int, float)) else "-"
    mp = raw.get("mp_ratio")
    mp_s = (
        f"{mp:.2f}"
        if raw.get("mp_visible", True) and isinstance(mp, (int, float))
        else "-"
    )
    params = " ".join(f"{k}={v}" for k, v in (action_obj.get("params") or {}).items())
    action = action_obj.get("name", "?") + (f"({params})" if params else "")
    parts = [f"tick {tick:06d}", f"scene={state.get('scene', '?')}",
             f"hp={hp_s}", f"mp={mp_s}", f"enemy_hp={enemy_s}"]
    if intent:
        parts.append(f"intent={intent}")
    parts.append(f"action={action}")
    return " ".join(parts)


def build_batch_prompt(
    frames: list,  # list[SampledFrame]
    records: list[dict],
    window_ticks: int,
    config_excerpt: str,
) -> str:
    """拼一批帧的复盘 prompt：每帧前后各 window_ticks tick 的操作日志 + 当前配置。"""
    sections = [
        "你是黑神话悟空自动战斗 AI 的复盘分析器。AI 通过截屏感知 + 战斗状态机 + "
        "覆盖式探索自动游戏（动作集：move/turn/light_attack/dodge/heal/lock_on）。",
        f"下面给你 {len(frames)} 个采样帧（图片已按下列顺序附上），"
        f"每帧附前后各 {window_ticks} tick 的状态/决策/操作日志，以及当前配置参数。",
        "当前配置（configs/wukong.yaml）:\n```yaml\n" + config_excerpt.strip() + "\n```",
    ]
    for frame in frames:
        lines = [f"=== 帧 {frame.path.name}（tick {frame.tick}）==="]
        window = tick_window(records, frame.tick, window_ticks)
        if window:
            lines.extend(format_record_line(i, rec) for i, rec in window)
        else:
            lines.append("（回放中无对应 tick 窗口记录）")
        sections.append("\n".join(lines))
    sections.append(
        "请分析：AI 的判断与操作是否合理（战斗状态机、探索漫游、避障转向）、"
        "是否存在异常（卡死原地转圈、反复无效动作、低血不喝药、脱战判断错误）、"
        "哪些配置参数应该调整、调整理由是什么。"
    )
    sections.append("只输出如下 JSON，不要输出任何其他内容:\n" + JSON_OUTPUT_SPEC)
    return "\n\n".join(sections)
