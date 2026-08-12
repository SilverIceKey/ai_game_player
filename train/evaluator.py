"""离线评估入口封装（spec §7：Candidate → Offline Evaluation → Promote/Reject）。

委托 evaluation/offline.py 的纯函数指标，通过注入评估函数保持松耦合：
- 默认使用 evaluation.offline.evaluate_samples（spec §35/§36）；
- 测试或实验可注入任意 (predictions, targets) -> dict 函数；
- 可选挂接 ModelRegistry，评估后按 spec §7 记录结果。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Protocol, Sequence

from capture.action import NormalizedAction
from evaluation.offline import evaluate_samples

if TYPE_CHECKING:
    from train.registry import ModelRegistry


class OfflineEvalFn(Protocol):
    """离线评估函数协议：预测序列 + 目标序列 → 指标 dict。"""

    def __call__(
        self,
        predictions: Sequence[NormalizedAction],
        targets: Sequence[NormalizedAction],
    ) -> dict[str, Any]: ...


class CandidateEvaluator:
    """候选模型离线评估入口。"""

    def __init__(self, eval_fn: OfflineEvalFn | None = None) -> None:
        self._eval_fn: OfflineEvalFn = eval_fn or evaluate_samples

    def evaluate(
        self,
        predictions: Sequence[NormalizedAction],
        targets: Sequence[NormalizedAction],
    ) -> dict[str, Any]:
        """执行离线评估，返回指标 dict（spec §36：禁止只看 overall accuracy）。"""
        return self._eval_fn(predictions, targets)

    def evaluate_and_record(
        self,
        registry: ModelRegistry,
        model_version: str,
        predictions: Sequence[NormalizedAction],
        targets: Sequence[NormalizedAction],
    ) -> dict[str, Any]:
        """评估并把结果记录进 registry（spec §7 流程的 Evaluate 环节）。"""
        result = self.evaluate(predictions, targets)
        registry.record_evaluation(model_version, result)
        return result
