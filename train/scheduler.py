"""训练时机调度（spec §6：训练只允许发生在 episode 结束 / 暂停阶段）。

游戏进行中只采集，禁止训练（与游戏抢 GPU、破坏采集时序）。
本类是纯逻辑状态机：

- `on_episode_end()`：episode 结束回调，登记一次待训练请求；
- `may_train(state)`：当前运行状态下是否允许开始训练；
- `notify_training_started()`：训练真正开始，消费待训练请求。

典型接线：dataset/episode.py 的 STOP EPISODE 回调 → on_episode_end()；
训练入口循环轮询 may_train(current_state)。
"""
from __future__ import annotations

# 运行状态（由 app 层上报；游戏中 = 禁止训练）
STATE_EPISODE_ACTIVE = "episode_active"  # episode 进行中（采集/控制中）
STATE_PAUSED = "paused"  # 暂停 / 加载阶段
STATE_EPISODE_ENDED = "episode_ended"  # episode 刚结束，未进入下一个
STATE_IDLE = "idle"  # 空闲（未开始 episode）

# 允许训练的状态：episode 结束 / 暂停 / 空闲（spec §6）
TRAINABLE_STATES: frozenset[str] = frozenset({STATE_PAUSED, STATE_EPISODE_ENDED, STATE_IDLE})

VALID_STATES: frozenset[str] = TRAINABLE_STATES | frozenset({STATE_EPISODE_ACTIVE})


class TrainScheduler:
    """决定何时允许训练（纯逻辑，无 IO）。"""

    def __init__(self) -> None:
        self._pending_episodes = 0

    @property
    def pending_episodes(self) -> int:
        """已结束但尚未触发训练的 episode 数。"""
        return self._pending_episodes

    def on_episode_end(self) -> None:
        """episode 结束回调：登记待训练请求。"""
        self._pending_episodes += 1

    def may_train(self, now_state: str) -> bool:
        """当前状态是否允许开始训练。

        条件：存在待训练请求，且当前不在 episode 进行中（spec §6）。
        """
        if now_state not in VALID_STATES:
            raise ValueError(f"未知运行状态: {now_state!r}（合法值: {sorted(VALID_STATES)}）")
        return self._pending_episodes > 0 and now_state in TRAINABLE_STATES

    def notify_training_started(self) -> None:
        """训练开始：消费一个待训练请求。"""
        if self._pending_episodes <= 0:
            raise RuntimeError("没有待训练请求，不能开始训练")
        self._pending_episodes -= 1
