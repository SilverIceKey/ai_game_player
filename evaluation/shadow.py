"""Shadow Mode 指标（spec §41）。

SHADOW 是 OBSERVE_TRAIN 的子状态：玩家仍然控制，AI 实时预测但不执行。
用于在进入 AUTOPILOT 前验证：
- 推理延迟（由 observability/metrics.py 负责）；
- Action 对齐（AI 想做的 vs 玩家实际做的，本模块）；
- 模型意图与关键动作召回（逐按钮 P/R）。

进入 AUTOPILOT 前必须通过 Shadow Mode（spec §41/§42 Phase 3）。
"""
from __future__ import annotations

from typing import Any

from capture.action import BUTTONS, NormalizedAction
from evaluation.offline import button_precision_recall


class ShadowMetrics:
    """SHADOW 模式下 AI 预测 vs 玩家操作的累计对齐指标。"""

    def __init__(self) -> None:
        self._count = 0
        self._move_sq_sum = 0.0
        self._camera_sq_sum = 0.0
        self._pred_buttons: list[frozenset[str]] = []
        self._actual_buttons: list[frozenset[str]] = []

    @property
    def count(self) -> int:
        return self._count

    def update(self, predicted: NormalizedAction, actual: NormalizedAction) -> None:
        """累计一对（AI 预测, 玩家实际）动作。"""
        self._count += 1
        self._move_sq_sum += (predicted.move_x - actual.move_x) ** 2 + (
            predicted.move_y - actual.move_y
        ) ** 2
        self._camera_sq_sum += (predicted.camera_x - actual.camera_x) ** 2 + (
            predicted.camera_y - actual.camera_y
        ) ** 2
        self._pred_buttons.append(predicted.buttons)
        self._actual_buttons.append(actual.buttons)

    def summary(self) -> dict[str, Any]:
        """汇总：样本数、move/camera 轴 MSE、逐按钮命中率（P/R）。"""
        n = self._count
        return {
            "sample_count": n,
            "move_mse": self._move_sq_sum / (2 * n) if n else 0.0,
            "camera_mse": self._camera_sq_sum / (2 * n) if n else 0.0,
            "buttons": button_precision_recall(self._pred_buttons, self._actual_buttons),
        }

    def render(self) -> str:
        """文本报告（session 结束打印/落盘用）。"""
        s = self.summary()
        lines = [
            f"shadow metrics (samples={s['sample_count']})",
            f"  move_mse   {s['move_mse']:.4f}",
            f"  camera_mse {s['camera_mse']:.4f}",
            f"  {'button':<14}{'precision':>10}{'recall':>8}{'tp':>6}{'fp':>6}{'fn':>6}",
        ]
        for name in BUTTONS:
            b = s["buttons"][name]
            if b["tp"] or b["fp"] or b["fn"]:
                lines.append(
                    f"  {name:<14}{b['precision']:>10.3f}{b['recall']:>8.3f}"
                    f"{b['tp']:>6}{b['fp']:>6}{b['fn']:>6}"
                )
        return "\n".join(lines)
