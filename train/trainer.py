"""Trainer 骨架（spec §23 Behavior Cloning + §6 训练时机禁令）。

spec §6 不允许的训练方式：采集时实时 backward()——与游戏抢 GPU、
破坏采集时序、产生 latency jitter、训练不可复现。训练只允许发生在
episode 结束 / 暂停 / 加载阶段（由 train/scheduler.py 控制时机）。

spec §23 Loss：

    L = L_move + λc·L_camera + λb·L_button + λt·L_temporal

所有 Loss 权重必须配置化（config.LossWeights）并记录进
ModelCheckpointMeta.training_config（spec §29 可复现性）。

本轮不引入 PyTorch：本类是流程骨架，`train_candidate()` 在无 torch
环境下抛出带指引的 RuntimeError。
"""
from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any, Protocol

from config import LossWeights
from model.checkpoint import ModelCheckpointMeta

if TYPE_CHECKING:
    from config import SamplingConfig


class SampleBuffer(Protocol):
    """Replay Buffer 最小协议（spec §28）。

    与 dataset/replay_buffer.py 的具体实现解耦：trainer 只依赖
    sample/size，测试中用假对象注入。
    """

    def sample(self, n: int, weights: Any = None) -> list[dict[str, Any]]:
        """按类别权重采样 n 个样本（样本结构见 spec §22）。"""
        ...

    def size(self) -> int:
        """当前 buffer 中样本总数。"""
        ...


_TORCH_MISSING_MSG = (
    "模型训练需要 PyTorch。本轮 trainer 为接口骨架，未引入 torch；"
    "安装 torch 并实现 spec §16-§19（模型架构/三阶段训练/拆 Head）"
    "与 §23（Behavior Cloning Loss）后重试。"
)


class Trainer:
    """候选模型训练器骨架。

    - replay buffer / loss 权重 / 训练配置全部构造注入；
    - 训练产物为 ModelCheckpointMeta（spec §29），状态流转交给
      train/registry.py；
    - 调用方必须先经 TrainScheduler.may_train() 确认时机（spec §6）。
    """

    def __init__(
        self,
        replay_buffer: SampleBuffer,
        loss_weights: LossWeights,
        training_config: dict[str, Any] | None = None,
        sampling_weights: SamplingConfig | None = None,
    ) -> None:
        self._buffer = replay_buffer
        self._loss_weights = loss_weights
        self._training_config: dict[str, Any] = dict(training_config or {})
        self._sampling_weights = sampling_weights

    @property
    def loss_weights(self) -> LossWeights:
        return self._loss_weights

    def train_candidate(
        self,
        dataset_version: str,
        model_version: str,
        code_commit: str = "",
    ) -> ModelCheckpointMeta:
        """在指定 dataset version 上训练候选模型（spec §7 Candidate Model）。

        无 torch 环境抛 RuntimeError；实现落地后返回记录了
        dataset_version / code_commit / training_config 的元数据。
        """
        if not dataset_version.strip():
            raise ValueError("dataset_version 不能为空（spec §29）")
        if not model_version.strip():
            raise ValueError("model_version 不能为空")
        if importlib.util.find_spec("torch") is None:
            raise RuntimeError(_TORCH_MISSING_MSG)
        raise RuntimeError(f"训练逻辑尚未实现：{_TORCH_MISSING_MSG}")
