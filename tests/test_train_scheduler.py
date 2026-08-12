"""train/scheduler.py 单元测试：训练时机纯逻辑（spec §6）。"""
from __future__ import annotations

import pytest

from train.scheduler import (
    STATE_EPISODE_ACTIVE,
    STATE_EPISODE_ENDED,
    STATE_IDLE,
    STATE_PAUSED,
    TrainScheduler,
)


def test_no_pending_episode_never_trainable() -> None:
    s = TrainScheduler()
    for state in (STATE_EPISODE_ACTIVE, STATE_PAUSED, STATE_EPISODE_ENDED, STATE_IDLE):
        assert s.may_train(state) is False


def test_episode_active_forbids_training() -> None:
    s = TrainScheduler()
    s.on_episode_end()
    assert s.may_train(STATE_EPISODE_ACTIVE) is False


def test_trainable_states_after_episode_end() -> None:
    s = TrainScheduler()
    s.on_episode_end()
    assert s.may_train(STATE_EPISODE_ENDED) is True
    assert s.may_train(STATE_PAUSED) is True
    assert s.may_train(STATE_IDLE) is True


def test_notify_training_started_consumes_pending() -> None:
    s = TrainScheduler()
    s.on_episode_end()
    s.on_episode_end()
    assert s.pending_episodes == 2
    s.notify_training_started()
    assert s.pending_episodes == 1
    assert s.may_train(STATE_IDLE) is True
    s.notify_training_started()
    assert s.may_train(STATE_IDLE) is False


def test_notify_without_pending_raises() -> None:
    with pytest.raises(RuntimeError, match="待训练"):
        TrainScheduler().notify_training_started()


def test_unknown_state_raises() -> None:
    s = TrainScheduler()
    s.on_episode_end()
    with pytest.raises(ValueError, match="未知运行状态"):
        s.may_train("playing")
