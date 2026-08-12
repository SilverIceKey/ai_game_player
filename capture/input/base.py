"""输入采集统一协议（spec §10：不同输入设备统一转换为 NormalizedAction）。

设计（重构计划 §3）：所有输入采集器（键鼠 / 手柄）统一为
「内部 queue + start/stop 线程模型」的拉取式接口：

- start()：启动底层监听/轮询线程
- stop()：停止并回收线程（幂等）
- poll(timeout)：取下一个 ActionRecord；超时或无事件返回 None

事件即 ActionRecord（capture.action），时间戳 = 事件发生的 now_us()
（spec §11：输入事件在回调/轮询收包时刻打点）。采集线程与消费线程
（Episode 记录/推理）通过 queue 解耦，禁止互相阻塞（spec §30）。
"""
from __future__ import annotations

import queue
from typing import Protocol, runtime_checkable

from capture.action import ActionRecord


@runtime_checkable
class InputCapture(Protocol):
    """输入采集器协议：start/stop 生命周期 + poll 拉取动作事件。"""

    def start(self) -> None:
        """启动采集（底层线程/钩子）。幂等。"""
        ...

    def stop(self) -> None:
        """停止采集并回收线程资源。幂等。"""
        ...

    def poll(self, timeout: float | None = None) -> ActionRecord | None:
        """取下一个动作事件；timeout 秒内无事件返回 None（timeout=None 无限等）。"""
        ...


class QueuedInputCapture:
    """InputCapture 的公共基类：内部 queue + poll 实现。

    子类负责在 start() 中启动事件来源（监听器/轮询线程），在事件发生时
    调用 _emit(record) 入队；事件时间戳必须在事件发生时刻打好再传入。
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[ActionRecord] = queue.Queue()

    def _emit(self, record: ActionRecord) -> None:
        self._queue.put(record)

    def poll(self, timeout: float | None = None) -> ActionRecord | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def start(self) -> None:  # pragma: no cover - 由子类实现
        raise NotImplementedError

    def stop(self) -> None:  # pragma: no cover - 由子类实现
        raise NotImplementedError
