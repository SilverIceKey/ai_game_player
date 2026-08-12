"""runtime/action_scheduler.py 单元测试（spec §15 Action Chunking / §26/§40 清空）。

纯逻辑：时间全部注入（微秒整数），不触达真实时钟。
"""
from __future__ import annotations

import pytest

from capture.action import ActionChunk, NormalizedAction
from runtime.action_scheduler import ActionScheduler

STEP_MS = 50.0
STEP_US = 50_000


def _chunk(n: int = 4) -> ActionChunk:
    # 每步 move_x 不同，便于断言吐出的是第几步
    return ActionChunk(
        actions=tuple(NormalizedAction(move_x=(i + 1) / 10) for i in range(n)),
        step_ms=STEP_MS,
        model_version="model-v001",
        created_us=0,
    )


class TestActionScheduler:
    def test_step_ms_must_be_positive(self):
        with pytest.raises(ValueError):
            ActionScheduler(0)

    def test_empty_returns_none(self):
        assert ActionScheduler(STEP_MS).due_action(0) is None

    def test_first_action_dispatched_immediately(self):
        sched = ActionScheduler(STEP_MS)
        chunk = _chunk()
        sched.submit_chunk(chunk)
        action = sched.due_action(1_000_000)
        assert action is chunk.actions[0]

    def test_pacing_by_step_ms(self):
        sched = ActionScheduler(STEP_MS)
        chunk = _chunk()
        sched.submit_chunk(chunk)
        t0 = 1_000_000
        assert sched.due_action(t0) is chunk.actions[0]
        # 同一个 step 内重复调用不再吐出
        assert sched.due_action(t0 + 10_000) is None
        assert sched.due_action(t0 + STEP_US - 1) is None
        # 到达下一个 step
        assert sched.due_action(t0 + STEP_US) is chunk.actions[1]
        assert sched.due_action(t0 + 2 * STEP_US) is chunk.actions[2]

    def test_skips_intermediate_steps_when_time_jumps(self):
        """调度抖动跨过多个 step 时直接取当前步，不慢放。"""
        sched = ActionScheduler(STEP_MS)
        chunk = _chunk()
        sched.submit_chunk(chunk)
        t0 = 1_000_000
        sched.due_action(t0)
        # 时间直接跳到第 3 步（index 2 之后）
        assert sched.due_action(t0 + 3 * STEP_US) is chunk.actions[3]

    def test_chunk_exhausted_returns_none_and_clears(self):
        sched = ActionScheduler(STEP_MS)
        sched.submit_chunk(_chunk(2))
        t0 = 1_000_000
        sched.due_action(t0)
        sched.due_action(t0 + STEP_US)
        assert sched.due_action(t0 + 2 * STEP_US) is None
        assert not sched.has_chunk  # 派发完自动清空，等待下一次推理
        assert sched.due_action(t0 + 3 * STEP_US) is None

    def test_submit_replaces_pending_chunk(self):
        """新 chunk 到达即作旧（spec §15：未执行的步丢弃）。"""
        sched = ActionScheduler(STEP_MS)
        old = _chunk()
        sched.submit_chunk(old)
        t0 = 1_000_000
        assert sched.due_action(t0) is old.actions[0]
        new = _chunk()
        sched.submit_chunk(new)
        # 新 chunk 从第 0 步重新开始，立即派发
        assert sched.due_action(t0 + 10_000) is new.actions[0]
        assert sched.due_action(t0 + 10_000 + STEP_US) is new.actions[1]

    def test_clear_stops_dispatch(self):
        """§26 接管 / §40 急停：clear 后不再吐出任何动作。"""
        sched = ActionScheduler(STEP_MS)
        sched.submit_chunk(_chunk())
        sched.due_action(1_000_000)
        sched.clear()
        assert not sched.has_chunk
        assert sched.due_action(1_000_000 + STEP_US) is None
