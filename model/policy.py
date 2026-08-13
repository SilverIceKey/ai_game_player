"""Video-Action Policy 协议与占位实现（spec §15/§16）。

模型本体本轮不引入 PyTorch：
- `VideoActionPolicy` 是推理侧契约，runtime 的 Inference Worker 只依赖本协议；
- `PlaceholderPolicy` 输出全 neutral 的 ActionChunk，用于 AUTOPILOT 链路验证（§42 Phase 0）；
- `RandomPolicy` 输出随机动作，用于管线调试（验证调度/安全过滤/日志通路）；
- `load_policy()` 是统一加载入口：无 checkpoint 时回退 Placeholder，
  有 checkpoint 但环境无 torch 时给出明确报错与指引。

输入契约（spec §8）：
- frames: Video History 窗口的 (frame, timestamp_us) 列表（预处理后的 np.ndarray
  配采集时间戳，旧→新；时间戳用于 token 年龄，spec §16 age bias）；
- action_history: 最近的 ActionRecord 列表（旧→新，时间戳同上）；
- audio_pcm: 与 Video History 对齐的过去窗口音频（float32 mono，spec §8.5，可选；
  带音频分支的 checkpoint 必传，TorchPolicy 内部转 log-mel）。

输出契约（spec §15 Action Chunking）：一次推理返回未来若干步动作。
"""
from __future__ import annotations

import importlib.util
import random
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from capture.action import ActionChunk, ActionRecord, BUTTONS, NormalizedAction
from capture.clock import now_us

if TYPE_CHECKING:
    import numpy as np

# spec §13/§15 默认值（与 config.PredictionConfig 默认一致）
DEFAULT_FUTURE_STEPS = 4
DEFAULT_STEP_MS = 50.0


@runtime_checkable
class VideoActionPolicy(Protocol):
    """端到端 Video-Action Policy 推理协议（spec §16）。"""

    @property
    def model_version(self) -> str:
        """当前模型版本标识（spec §29，写入推理日志 §33）。"""
        ...

    def predict(
        self,
        frames: list[tuple[np.ndarray, int]],
        action_history: list[ActionRecord],
        audio_pcm: np.ndarray | None = None,
    ) -> ActionChunk:
        """由 Video History + Action History（+ 可选 Audio History，§8.5）预测未来 Action Chunk。"""
        ...


class PlaceholderPolicy:
    """占位 Policy：恒输出全 neutral 动作（等效松开所有输入）。

    用于在真实模型落地前验证 AUTOPILOT 全链路（spec §42 Phase 0）。
    """

    model_version = "placeholder"

    def __init__(
        self,
        future_action_steps: int = DEFAULT_FUTURE_STEPS,
        step_ms: float = DEFAULT_STEP_MS,
    ) -> None:
        if future_action_steps <= 0:
            raise ValueError(f"future_action_steps 必须为正整数: {future_action_steps}")
        if step_ms <= 0:
            raise ValueError(f"step_ms 必须为正数: {step_ms}")
        self._future_action_steps = future_action_steps
        self._step_ms = float(step_ms)

    def predict(
        self,
        frames: list[tuple[np.ndarray, int]],
        action_history: list[ActionRecord],
        audio_pcm: np.ndarray | None = None,
    ) -> ActionChunk:
        return ActionChunk(
            actions=tuple(NormalizedAction.neutral() for _ in range(self._future_action_steps)),
            step_ms=self._step_ms,
            model_version=self.model_version,
            confidence={"neutral": 1.0},
            created_us=now_us(),
        )


class RandomPolicy:
    """随机 Policy：随机连续轴 + 随机按钮子集，seed 可注入保证可复现。

    仅用于管线调试（动作调度、安全过滤、日志字段通路），不是训练基线。
    """

    model_version = "random"

    def __init__(
        self,
        seed: int | None = None,
        future_action_steps: int = DEFAULT_FUTURE_STEPS,
        step_ms: float = DEFAULT_STEP_MS,
    ) -> None:
        if future_action_steps <= 0:
            raise ValueError(f"future_action_steps 必须为正整数: {future_action_steps}")
        if step_ms <= 0:
            raise ValueError(f"step_ms 必须为正数: {step_ms}")
        self._rng = random.Random(seed)
        self._future_action_steps = future_action_steps
        self._step_ms = float(step_ms)

    def _random_action(self) -> NormalizedAction:
        rng = self._rng
        buttons = frozenset(name for name in BUTTONS if rng.random() < 0.15)
        return NormalizedAction(
            move_x=rng.uniform(-1.0, 1.0),
            move_y=rng.uniform(-1.0, 1.0),
            camera_x=rng.uniform(-1.0, 1.0),
            camera_y=rng.uniform(-1.0, 1.0),
            buttons=buttons,
        )

    def predict(
        self,
        frames: list[tuple[np.ndarray, int]],
        action_history: list[ActionRecord],
        audio_pcm: np.ndarray | None = None,
    ) -> ActionChunk:
        return ActionChunk(
            actions=tuple(self._random_action() for _ in range(self._future_action_steps)),
            step_ms=self._step_ms,
            model_version=self.model_version,
            confidence={},
            created_us=now_us(),
        )


def load_policy(checkpoint_path: str | Path | None = None) -> VideoActionPolicy:
    """加载 Policy：无 checkpoint 返回 PlaceholderPolicy。

    有 checkpoint 路径（checkpoints/<version>/ 目录或其中文件）时加载真实
    TorchPolicy；环境无 torch 时抛出带指引的 RuntimeError，而不是静默回退占位模型。
    """
    if checkpoint_path is None:
        return PlaceholderPolicy()

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {path}")
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError(
            f"加载 checkpoint 需要 PyTorch（{path}）：pip install torch torchvision；"
            "无 checkpoint 时 load_policy(None) 返回 PlaceholderPolicy 用于链路验证。"
        )
    from model.torch_policy import load_torch_policy

    return load_torch_policy(path)
