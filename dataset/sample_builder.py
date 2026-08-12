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

from capture.action import ActionRecord


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
) -> list[dict[str, Any]]:
    """构造样本列表。frames/actions 必须按 timestamp_us 升序。

    返回的每个样本 dict：
    - anchor_us: anchor 时刻
    - frames: 按时间从旧到新，每项 {ref, target_us, timestamp_us, deviation_us}
    - action_history: 最近 history_actions 条 ≤ anchor 的 ActionRecord（可能更少）
    - target_actions: 每项 {record, label_us, timestamp_us, deviation_us}
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

    grid_step_us = round(1_000_000 / params.sample_fps)
    action_step_us = round(params.action_step_ms * 1000)
    offset_us = round(params.action_label_offset_ms * 1000)

    samples: list[dict[str, Any]] = []
    anchor = frame_ts[0]
    while anchor <= frame_ts[-1]:
        sample = _build_one(
            anchor, frames, frame_ts, actions, action_ts,
            params, grid_step_us, action_step_us, offset_us,
        )
        if sample is not None:
            samples.append(sample)
        anchor += grid_step_us
    return samples


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
        targets.append(
            {
                "record": actions[idx],
                "label_us": label_us,
                "timestamp_us": action_ts[idx],
                "deviation_us": label_us - action_ts[idx],
            }
        )

    return {
        "anchor_us": anchor_us,
        "frames": obs,
        "action_history": history,
        "target_actions": targets,
    }


def _check_sorted(timestamps: list[int], name: str) -> None:
    for i in range(1, len(timestamps)):
        if timestamps[i] < timestamps[i - 1]:
            raise ValueError(
                f"{name} 必须按 timestamp_us 升序: 下标 {i - 1} ({timestamps[i - 1]})"
                f" > 下标 {i} ({timestamps[i]})"
            )
