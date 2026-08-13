"""训练样本构造（spec §22 Training Sample + §12 Human Reaction Delay）。

从 Episode Store 的帧索引与动作记录构造训练样本：

```text
Observation:     frames[t-k..t]    k = history_frames，按 sample_fps 网格回溯
Action History:  最近 m 条 ≤ t 的 ActionRecord（m = history_actions）
Target:          未来 future_action_steps 步，每步 action_step_ms，
                 标签时刻整体偏移 action_label_offset_ms（§12 人类反应延迟补偿）
```

时间对齐策略（spec §11 目标同步误差 <10ms）：
- 对每个目标时刻，一律取"时间戳不超过该时刻的最近一条记录"（向后因果，不取未来数据）。
- 每个采样槽位都记录实际偏差 deviation_us = 目标时刻 - 实际记录时刻（恒 ≥ 0），
  供训练侧统计真实同步误差，而不是假设完美对齐。
- 帧图像不在这里加载：样本只保存帧引用 ref（由调用方解释，如
  (video_path, video_offset)），核心时间对齐逻辑无视频文件即可测试。

跳过规则（显式约定）：
- 观测窗口最老槽位之前没有帧（样本开头帧不足）→ 跳过该 anchor。
- 某未来步的标签时刻之前没有任何动作记录（采集还没开始）→ 跳过该 anchor。
- 标签时刻超过最后一条动作记录：不跳过，动作状态持续有效，偏差如实记录。
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Sequence

from capture.action import SOURCE_AI, ActionRecord

# spec §26 数据闭环：样本段标签（由 telemetry marker 推导，采集层不写）
SEGMENT_HUMAN_DEMONSTRATION = "human_demonstration"  # OBSERVE_TRAIN 人类示范
SEGMENT_HUMAN_CORRECTION = "human_correction"  # 接管段内：人类实际动作（DAgger 纠正）
SEGMENT_AUTOPILOT_SUCCESS = "autopilot_success"  # AI 控制且未临近接管
SEGMENT_AUTOPILOT_FAILURE = "autopilot_failure"  # 接管前窗口：AI 动作不得回灌为 imitation target

_MARKER_OVERRIDE_START = "HUMAN_OVERRIDE_START"
_MARKER_AUTOPILOT_RESUME = "AUTOPILOT_RESUME"


@dataclass(frozen=True)
class FrameStamp:
    """帧索引条目：时间戳 + 不透明引用（由帧加载方解释）。"""

    timestamp_us: int
    ref: Any


@dataclass(frozen=True)
class SampleParams:
    """样本构造参数（对应 config 的 model/prediction/labels 三段）。"""

    sample_fps: float = 12.0  # spec §13
    history_frames: int = 16  # spec §13/§8.1
    history_actions: int = 16  # spec §8.2
    action_step_ms: float = 50.0  # spec §13/§15
    future_action_steps: int = 4  # spec §13/§15
    action_label_offset_ms: float = 0.0  # spec §12，实验搜索 0~250


def build_samples(
    frames: Sequence[FrameStamp],
    actions: Sequence[ActionRecord],
    params: SampleParams,
    markers: Sequence[dict[str, Any]] | None = None,
    pre_override_window_us: int = 2_000_000,
    stats: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """构造样本列表。frames/actions 必须按 timestamp_us 升序。

    返回的每个样本 dict：
    - anchor_us: anchor 时刻
    - frames: 按时间从旧到新，每项 {ref, target_us, timestamp_us, deviation_us}
    - action_history: 最近 history_actions 条 ≤ anchor 的 ActionRecord（可能更少）
    - target_actions: 每项 {record, label_us, timestamp_us, deviation_us}
    - segment: 段标签（human_demonstration / human_correction / autopilot_success /
      autopilot_failure，spec §26 数据闭环）

    markers（telemetry 中 type=="marker" 的事件）存在时启用段推导与 target 过滤：
    - 接管段 [OVERRIDE_START, RESUME] → human_correction（target 来自人类操作）
    - START 前 pre_override_window → autopilot_failure，其中 AI 动作不可作 target
      （导致该 anchor 被跳过，stats["skipped_autopilot_failure"] 计数）
    - 其余 → autopilot_success，AI 动作可作 target（成功段自模仿）
    markers 为空/None（OBSERVE_TRAIN）→ 全部 human_demonstration，行为不变。
    """
    if params.sample_fps <= 0:
        raise ValueError(f"sample_fps 必须为正: {params.sample_fps}")
    if params.history_frames <= 0 or params.future_action_steps <= 0:
        raise ValueError("history_frames / future_action_steps 必须为正整数")
    if not frames:
        return []

    frame_ts = [f.timestamp_us for f in frames]
    action_ts = [a.timestamp_us for a in actions]
    _check_sorted(frame_ts, "frames")
    _check_sorted(action_ts, "actions")

    classifier = _SegmentClassifier(markers, pre_override_window_us)

    grid_step_us = round(1_000_000 / params.sample_fps)
    action_step_us = round(params.action_step_ms * 1000)
    offset_us = round(params.action_label_offset_ms * 1000)

    samples: list[dict[str, Any]] = []
    anchor = frame_ts[0]
    while anchor <= frame_ts[-1]:
        sample = _build_one(
            anchor, frames, frame_ts, actions, action_ts,
            params, grid_step_us, action_step_us, offset_us, classifier,
        )
        if sample is not None:
            samples.append(sample)
        anchor += grid_step_us
    if stats is not None:
        stats["skipped_autopilot_failure"] = classifier.skipped_failure
    return samples


class _SegmentClassifier:
    """由 marker 事件推导时间线段落（spec §26：采集层不写段标签，构建期推导）。

    classify(t) ∈ {"override", "pre", "post", "ai_control"}；无 markers 时
    enabled=False，调用方走 human_demonstration 旧路径。
    """

    def __init__(self, markers: Sequence[dict[str, Any]] | None, window_us: int):
        self.enabled = bool(markers)
        self._window_us = int(window_us)
        self.skipped_failure = 0
        self._intervals: list[tuple[int, int | None]] = []  # (start, end|None=未闭合)
        open_start: int | None = None
        for event in sorted(markers or [], key=lambda e: int(e["timestamp_us"])):
            marker = event.get("marker")
            ts = int(event["timestamp_us"])
            if marker == _MARKER_OVERRIDE_START and open_start is None:
                open_start = ts
            elif marker == _MARKER_AUTOPILOT_RESUME and open_start is not None:
                self._intervals.append((open_start, ts))
                open_start = None
        if open_start is not None:  # session 结束于接管中
            self._intervals.append((open_start, None))

    def classify(self, t_us: int) -> str:
        for start, end in self._intervals:
            if start <= t_us and (end is None or t_us <= end):
                return "override"
            if start - self._window_us <= t_us < start:
                return "pre"
            if end is not None and end < t_us <= end + self._window_us:
                return "post"
        return "ai_control"

    def segment_of(self, anchor_us: int) -> str:
        zone = self.classify(anchor_us)
        if zone == "override":
            return SEGMENT_HUMAN_CORRECTION
        if zone == "pre":
            return SEGMENT_AUTOPILOT_FAILURE
        return SEGMENT_AUTOPILOT_SUCCESS  # post 段归入 success（恢复后 AI 正常控制）

    def ai_target_usable(self, t_us: int) -> bool:
        """AI 来源动作能否作 imitation target：pre/override 段内不可用（spec §26）。"""
        return self.classify(t_us) in ("ai_control", "post")


def _build_one(
    anchor_us: int,
    frames: Sequence[FrameStamp],
    frame_ts: list[int],
    actions: Sequence[ActionRecord],
    action_ts: list[int],
    params: SampleParams,
    grid_step_us: int,
    action_step_us: int,
    offset_us: int,
    classifier: _SegmentClassifier,
) -> dict[str, Any] | None:
    # ---- Observation：窗口内按 sample_fps 间隔回溯，逐槽取 ≤ 目标时刻的最近帧 ----
    obs: list[dict[str, Any]] = []
    for back in range(params.history_frames - 1, -1, -1):
        target_us = anchor_us - back * grid_step_us
        idx = bisect_right(frame_ts, target_us) - 1
        if idx < 0:
            return None  # 窗口最老槽位之前没有帧：开头帧不足，跳过
        obs.append(
            {
                "ref": frames[idx].ref,
                "target_us": target_us,
                "timestamp_us": frame_ts[idx],
                "deviation_us": target_us - frame_ts[idx],
            }
        )

    # ---- Action History：最近 history_actions 条 ≤ anchor ----
    hi = bisect_right(action_ts, anchor_us)
    history = list(actions[max(0, hi - params.history_actions) : hi])

    # ---- Target：未来步 + §12 标签偏移，逐槽取 ≤ 标签时刻的最近动作 ----
    targets: list[dict[str, Any]] = []
    for step in range(1, params.future_action_steps + 1):
        label_us = anchor_us + step * action_step_us + offset_us
        idx = bisect_right(action_ts, label_us) - 1
        if idx < 0:
            return None  # 标签时刻之前还没有任何动作记录，无法标注，跳过
        record = actions[idx]
        # spec §26 数据闭环：AI 来源动作在 pre/override 段内不得作 imitation target
        # （失败前的 AI 动作不回灌）；该 anchor 整样本跳过并计数
        if (
            classifier.enabled
            and record.source == SOURCE_AI
            and not classifier.ai_target_usable(label_us)
        ):
            classifier.skipped_failure += 1
            return None
        targets.append(
            {
                "record": record,
                "label_us": label_us,
                "timestamp_us": action_ts[idx],
                "deviation_us": label_us - action_ts[idx],
            }
        )

    # spec §26 数据闭环：pre（接管前窗口）段 anchor 整段剔除——无论 target 来源，
    # 失败临近段的观测上下文不作为训练样本；AI 来源动作在 pre/override 段内
    # 不得作 imitation target（失败前的 AI 动作不回灌），该 anchor 跳过并计数
    if classifier.enabled and classifier.classify(anchor_us) == "pre":
        classifier.skipped_failure += 1
        return None
    return {
        "anchor_us": anchor_us,
        "frames": obs,
        "action_history": history,
        "target_actions": targets,
        "segment": (
            classifier.segment_of(anchor_us)
            if classifier.enabled
            else SEGMENT_HUMAN_DEMONSTRATION
        ),
    }


def _check_sorted(timestamps: list[int], name: str) -> None:
    for i in range(1, len(timestamps)):
        if timestamps[i] < timestamps[i - 1]:
            raise ValueError(
                f"{name} 必须按 timestamp_us 升序: 下标 {i - 1} ({timestamps[i - 1]})"
                f" > 下标 {i} ({timestamps[i]})"
            )
