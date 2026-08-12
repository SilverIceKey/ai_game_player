"""离线评估指标（spec §35 固定 Eval Set + §36 Offline Metrics）。

spec §35：训练集不能覆盖固定冻结 Eval Set，类别见 EVAL_CATEGORIES。
spec §36：至少输出 Movement Error / Camera Error / 关键按钮 Precision / Recall，
禁止只看 Overall Accuracy（类别不平衡下该指标无意义，§24）。

本模块全部为纯函数，不依赖 torch / 具体模型。
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

from capture.action import BUTTONS, NormalizedAction

# spec §35 固定 Eval Set 类别（训练集禁止覆盖）
EVAL_CATEGORIES: tuple[str, ...] = (
    "traversal",
    "narrow_path",
    "interaction",
    "collection",
    "normal_combat",
    "low_hp",
    "boss_combo",
    "camera_lost",
    "cornered",
    "healing",
    "recovery",
    "death_restart",
)


def movement_error(pred: NormalizedAction, target: NormalizedAction) -> float:
    """Movement MSE：move_x / move_y 两轴平方误差的均值。"""
    return ((pred.move_x - target.move_x) ** 2 + (pred.move_y - target.move_y) ** 2) / 2.0


def camera_error(pred: NormalizedAction, target: NormalizedAction) -> float:
    """Camera MSE：camera_x / camera_y 两轴平方误差的均值。"""
    return ((pred.camera_x - target.camera_x) ** 2 + (pred.camera_y - target.camera_y) ** 2) / 2.0


def button_precision_recall(
    pred_buttons_seq: Sequence[Iterable[str]],
    target_buttons_seq: Sequence[Iterable[str]],
) -> dict[str, dict[str, float]]:
    """逐按钮 Precision / Recall（spec §36 关键动作指标）。

    输入为等长的按钮名集合序列（每步按下的按钮）。
    返回 {按钮: {"precision", "recall", "tp", "fp", "fn"}}；
    分母为 0 时对应指标记 0.0（无预测/无目标，无从谈起）。
    """
    if len(pred_buttons_seq) != len(target_buttons_seq):
        raise ValueError(
            f"预测与目标序列长度不一致: {len(pred_buttons_seq)} vs {len(target_buttons_seq)}"
        )
    valid = set(BUTTONS)
    tp = dict.fromkeys(BUTTONS, 0)
    fp = dict.fromkeys(BUTTONS, 0)
    fn = dict.fromkeys(BUTTONS, 0)
    for pred_set_raw, target_set_raw in zip(pred_buttons_seq, target_buttons_seq, strict=True):
        pred_set = set(pred_set_raw)
        target_set = set(target_set_raw)
        unknown = (pred_set | target_set) - valid
        if unknown:
            raise ValueError(f"未知按钮名: {sorted(unknown)}（合法值见 capture.action.BUTTONS）")
        for name in BUTTONS:
            if name in pred_set and name in target_set:
                tp[name] += 1
            elif name in pred_set:
                fp[name] += 1
            elif name in target_set:
                fn[name] += 1
    out: dict[str, dict[str, float]] = {}
    for name in BUTTONS:
        t, f_p, f_n = tp[name], fp[name], fn[name]
        out[name] = {
            "precision": t / (t + f_p) if (t + f_p) else 0.0,
            "recall": t / (t + f_n) if (t + f_n) else 0.0,
            "tp": t,
            "fp": f_p,
            "fn": f_n,
        }
    return out


def evaluate_samples(
    predictions: Sequence[NormalizedAction],
    targets: Sequence[NormalizedAction],
) -> dict[str, Any]:
    """汇总离线指标（spec §36）。

    返回：
    - sample_count
    - movement_error / camera_error（全样本均值）
    - buttons：逐按钮 Precision / Recall
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"预测与目标样本数不一致: {len(predictions)} vs {len(targets)}"
        )
    n = len(predictions)
    move_mse = sum(movement_error(p, t) for p, t in zip(predictions, targets, strict=True)) / n if n else 0.0
    cam_mse = sum(camera_error(p, t) for p, t in zip(predictions, targets, strict=True)) / n if n else 0.0
    return {
        "sample_count": n,
        "movement_error": move_mse,
        "camera_error": cam_mse,
        "buttons": button_precision_recall(
            [p.buttons for p in predictions],
            [t.buttons for t in targets],
        ),
    }
