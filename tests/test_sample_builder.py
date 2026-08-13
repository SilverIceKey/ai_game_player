"""dataset/sample_builder.py 单元测试：合成时间戳验证时间对齐/偏移/窗口边界/§26 段分类。

全程不触达视频文件（帧引用用字符串代替）。
"""
from __future__ import annotations

import pytest

from capture.action import (
    SOURCE_AI,
    SOURCE_CORRECTION,
    ActionRecord,
    NormalizedAction,
)
from dataset.sample_builder import (
    SEGMENT_AUTOPILOT_SUCCESS,
    SEGMENT_HUMAN_CORRECTION,
    SEGMENT_HUMAN_DEMONSTRATION,
    FrameStamp,
    SampleParams,
    build_samples,
)

T0 = 1_000_000
GRID_US = 10_000  # sample_fps=100 → 10ms 网格，方便手算


def _params(**kwargs) -> SampleParams:
    defaults = dict(
        sample_fps=100.0,
        history_frames=4,
        history_actions=3,
        action_step_ms=10.0,  # 与网格同长，便于手算
        future_action_steps=2,
        action_label_offset_ms=0.0,
    )
    defaults.update(kwargs)
    return SampleParams(**defaults)


def _frames(n: int, start: int = T0, step: int = GRID_US) -> list[FrameStamp]:
    return [FrameStamp(start + i * step, ref=f"f{i}") for i in range(n)]


def _actions(n: int, start: int = T0, step: int = GRID_US) -> list[ActionRecord]:
    return [
        ActionRecord(start + i * step, NormalizedAction(move_y=0.5 if i % 2 else 0.0))
        for i in range(n)
    ]


def test_aligned_grid_exact_match():
    frames = _frames(20)
    actions = _actions(30)
    samples = build_samples(frames, actions, _params())

    # 开头 history_frames-1=3 个 anchor 因观测窗口不足被跳过
    assert len(samples) == 17
    first = samples[0]
    assert first["anchor_us"] == T0 + 3 * GRID_US
    # 观测窗口从旧到新 f0..f3，网格完全对齐时偏差为 0
    assert [f["ref"] for f in first["frames"]] == ["f0", "f1", "f2", "f3"]
    assert all(f["deviation_us"] == 0 for f in first["frames"])
    # Action History：最近 3 条 ≤ anchor
    assert [a.timestamp_us for a in first["action_history"]] == [
        T0 + GRID_US,
        T0 + 2 * GRID_US,
        T0 + 3 * GRID_US,
    ]
    # Target：未来 2 步，每步 10ms
    assert [t["record"].timestamp_us for t in first["target_actions"]] == [
        T0 + 4 * GRID_US,
        T0 + 5 * GRID_US,
    ]
    assert all(t["deviation_us"] == 0 for t in first["target_actions"])


def test_label_offset_shifts_target():
    frames = _frames(20)
    actions = _actions(40)
    # §12：标签整体后移 50ms（= 5 个网格）
    samples = build_samples(frames, actions, _params(action_label_offset_ms=50.0))
    first = samples[0]
    assert [t["label_us"] for t in first["target_actions"]] == [
        T0 + 4 * GRID_US + 50_000,
        T0 + 5 * GRID_US + 50_000,
    ]
    assert [t["record"].timestamp_us for t in first["target_actions"]] == [
        T0 + 9 * GRID_US,
        T0 + 10 * GRID_US,
    ]
    # 观测与动作历史不受 offset 影响
    assert [f["ref"] for f in first["frames"]] == ["f0", "f1", "f2", "f3"]


def test_frame_deviation_recorded():
    # 帧不在网格上：槽位取"≤ 目标时刻的最近帧"，偏差如实记录
    frames = [
        FrameStamp(T0, ref="a"),
        FrameStamp(T0 + 8_000, ref="b"),
        FrameStamp(T0 + 12_000, ref="c"),
    ]
    actions = _actions(10)
    params = _params(history_frames=2)
    samples = build_samples(frames, actions, params)

    assert len(samples) == 1  # anchor=T0+10ms（T0 处窗口不足被跳过）
    obs = samples[0]["frames"]
    assert [f["ref"] for f in obs] == ["a", "b"]
    assert obs[0]["deviation_us"] == 0  # 目标 T0 精确命中
    assert obs[1]["deviation_us"] == 2_000  # 目标 T0+10ms，实际帧 T0+8ms


def test_insufficient_head_frames_skipped():
    frames = _frames(6)  # 6 帧，窗口 4 → 只有 3 个 anchor 有效
    actions = _actions(20)
    samples = build_samples(frames, actions, _params())
    assert [s["anchor_us"] for s in samples] == [
        T0 + 3 * GRID_US,
        T0 + 4 * GRID_US,
        T0 + 5 * GRID_US,
    ]


def test_action_history_window_and_short_head():
    frames = _frames(10)
    actions = _actions(10)
    samples = build_samples(frames, actions, _params(history_actions=2))
    # anchor=T0+30ms：取最近 2 条
    assert [a.timestamp_us for a in samples[0]["action_history"]] == [
        T0 + 2 * GRID_US,
        T0 + 3 * GRID_US,
    ]
    # 动作历史不足时不补 padding，如实给出（actions 从 T0 开始，anchor=T0+30ms 恰有 4 条可取 2 条）


def test_skip_when_no_action_before_label():
    # 动作采集开始得晚：所有标签时刻之前都没有动作记录 → 全部跳过
    frames = _frames(10)
    actions = _actions(5, start=T0 + 500_000)
    assert build_samples(frames, actions, _params()) == []


def test_action_state_persists_beyond_last_record():
    # 标签时刻超过最后一条动作记录：不跳过，状态持续，偏差如实记录
    frames = _frames(10)
    actions = [ActionRecord(T0, NormalizedAction(move_y=1.0))]  # 只有一条
    samples = build_samples(frames, actions, _params())
    assert len(samples) == 7
    last = samples[-1]
    assert last["anchor_us"] == T0 + 9 * GRID_US
    assert [t["record"].timestamp_us for t in last["target_actions"]] == [T0, T0]
    assert last["target_actions"][0]["deviation_us"] == 10 * GRID_US  # label=T0+100ms


def test_empty_frames_returns_empty():
    assert build_samples([], _actions(5), _params()) == []


def test_unsorted_input_raises():
    frames = [FrameStamp(T0 + GRID_US, ref="b"), FrameStamp(T0, ref="a")]
    with pytest.raises(ValueError, match="升序"):
        build_samples(frames, [], _params())
    actions = [ActionRecord(T0 + GRID_US, NormalizedAction()), ActionRecord(T0, NormalizedAction())]
    with pytest.raises(ValueError, match="升序"):
        build_samples(_frames(10), actions, _params())


def test_invalid_params_raise():
    with pytest.raises(ValueError, match="sample_fps"):
        build_samples(_frames(5), _actions(5), _params(sample_fps=0.0))
    with pytest.raises(ValueError, match="正整数"):
        build_samples(_frames(5), _actions(5), _params(history_frames=0))


# ---------- §26 数据闭环：段分类与失败段排除 ----------

OVERRIDE_START_US = T0 + 300_000
RESUME_US = T0 + 400_000
PRE_WINDOW_US = 50_000
_MARKERS = [
    {"type": "marker", "marker": "HUMAN_OVERRIDE_START", "timestamp_us": OVERRIDE_START_US},
    {"type": "marker", "marker": "AUTOPILOT_RESUME", "timestamp_us": RESUME_US},
]


def _mixed_actions() -> list[ActionRecord]:
    """AI 动作贯穿全程；接管段 [300ms, 400ms] 内叠 correction 记录。"""
    records = [
        ActionRecord(T0 + i * GRID_US, NormalizedAction(move_y=0.5), SOURCE_AI)
        for i in range(60)
    ]
    records += [
        ActionRecord(OVERRIDE_START_US + i * GRID_US, NormalizedAction(move_y=-1.0), SOURCE_CORRECTION)
        for i in range(10)
    ]
    return sorted(records, key=lambda r: r.timestamp_us)


def _build_with_markers(stats: dict | None = None):
    return build_samples(
        _frames(60), _mixed_actions(), _params(),
        markers=_MARKERS, pre_override_window_us=PRE_WINDOW_US, stats=stats,
    )


def test_segments_classified():
    samples = _build_with_markers()
    by_anchor = {s["anchor_us"]: s["segment"] for s in samples}
    # AI 控制段（远离接管窗口）→ autopilot_success
    assert by_anchor[T0 + 100_000] == SEGMENT_AUTOPILOT_SUCCESS
    # 接管段内 → human_correction，target 来自人类操作
    override_samples = [
        s for s in samples
        if OVERRIDE_START_US <= s["anchor_us"] <= RESUME_US - 2 * GRID_US
    ]
    assert override_samples
    assert all(s["segment"] == SEGMENT_HUMAN_CORRECTION for s in override_samples)
    assert all(
        t["record"].source == SOURCE_CORRECTION
        for s in override_samples
        for t in s["target_actions"]
    )


def test_failure_segment_ai_targets_excluded():
    """接管前窗口（pre 段）的 AI 动作不得回灌为 imitation target：相关 anchor 整样本跳过。"""
    stats: dict = {}
    samples = _build_with_markers(stats)
    assert stats["skipped_autopilot_failure"] > 0
    # pre 段 [250ms, 300ms) 内没有任何样本（target 必落在 pre/override 的 AI 动作上）
    assert not any(
        OVERRIDE_START_US - PRE_WINDOW_US <= s["anchor_us"] < OVERRIDE_START_US
        for s in samples
    )
    # 成功段 AI target 保留（自模仿）
    success = [s for s in samples if s["segment"] == SEGMENT_AUTOPILOT_SUCCESS]
    assert any(
        t["record"].source == SOURCE_AI for s in success for t in s["target_actions"]
    )


def test_no_markers_keeps_legacy_behavior():
    """无 markers（OBSERVE_TRAIN / 旧数据）：全部 human_demonstration，不过滤。"""
    samples = build_samples(_frames(20), _mixed_actions()[:30], _params())
    assert samples
    assert all(s["segment"] == SEGMENT_HUMAN_DEMONSTRATION for s in samples)
