"""evaluation/offline.py 单元测试：手算数值校验（spec §35/§36）。"""
from __future__ import annotations

import pytest

from capture.action import BUTTONS, NormalizedAction
from evaluation.offline import (
    EVAL_CATEGORIES,
    button_precision_recall,
    camera_error,
    evaluate_samples,
    movement_error,
)


def test_eval_categories_match_spec() -> None:
    assert EVAL_CATEGORIES == (
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


def test_movement_error_hand_computed() -> None:
    pred = NormalizedAction(move_x=1.0, move_y=-1.0)
    target = NormalizedAction(move_x=0.5, move_y=0.0)
    # ((1.0-0.5)^2 + (-1.0-0.0)^2) / 2 = (0.25 + 1.0) / 2 = 0.625
    assert movement_error(pred, target) == pytest.approx(0.625)
    assert movement_error(pred, pred) == 0.0


def test_camera_error_hand_computed() -> None:
    pred = NormalizedAction(camera_x=0.8, camera_y=-0.2)
    target = NormalizedAction(camera_x=0.0, camera_y=0.0)
    # (0.64 + 0.04) / 2 = 0.34
    assert camera_error(pred, target) == pytest.approx(0.34)


def test_button_precision_recall_hand_computed() -> None:
    pred = [{"dodge"}, {"attack_light"}, set()]
    target = [{"dodge"}, set(), {"heal"}]
    pr = button_precision_recall(pred, target)
    # dodge: tp=1 fp=0 fn=0 → P=1 R=1
    assert pr["dodge"]["precision"] == 1.0
    assert pr["dodge"]["recall"] == 1.0
    # attack_light: tp=0 fp=1 → P=0 R=0（fn=0 → recall 0.0 by convention）
    assert pr["attack_light"]["precision"] == 0.0
    assert pr["attack_light"]["fp"] == 1
    # heal: fn=1 → R=0；tp+fp=0 → P=0.0
    assert pr["heal"]["recall"] == 0.0
    assert pr["heal"]["fn"] == 1
    # 未出现的按钮全零
    assert pr["jump"] == {"precision": 0.0, "recall": 0.0, "tp": 0, "fp": 0, "fn": 0}
    # 覆盖全部按钮键
    assert set(pr.keys()) == set(BUTTONS)


def test_button_precision_recall_length_mismatch() -> None:
    with pytest.raises(ValueError, match="长度不一致"):
        button_precision_recall([{"dodge"}], [])


def test_button_precision_recall_unknown_button() -> None:
    with pytest.raises(ValueError, match="未知按钮名"):
        button_precision_recall([{"fly"}], [set()])


def test_evaluate_samples_hand_computed() -> None:
    predictions = [
        NormalizedAction(move_x=1.0, move_y=0.0, camera_x=0.0, camera_y=1.0, buttons={"dodge"}),
        NormalizedAction(move_x=0.0, move_y=0.0, camera_x=0.0, camera_y=0.0, buttons=set()),
    ]
    targets = [
        NormalizedAction(move_x=0.0, move_y=0.0, camera_x=0.0, camera_y=0.0, buttons={"dodge"}),
        NormalizedAction(move_x=0.0, move_y=1.0, camera_x=0.0, camera_y=0.0, buttons={"heal"}),
    ]
    result = evaluate_samples(predictions, targets)
    assert result["sample_count"] == 2
    # move: (1.0^2+0)/2=0.5, (0+1.0^2)/2=0.5 → 均值 0.5
    assert result["movement_error"] == pytest.approx(0.5)
    # camera: (0+1.0^2)/2=0.5, 0 → 均值 0.25
    assert result["camera_error"] == pytest.approx(0.25)
    # dodge tp=1 → P=R=1；heal fn=1 → R=0
    assert result["buttons"]["dodge"]["recall"] == 1.0
    assert result["buttons"]["heal"]["recall"] == 0.0


def test_evaluate_samples_empty() -> None:
    result = evaluate_samples([], [])
    assert result["sample_count"] == 0
    assert result["movement_error"] == 0.0
    assert result["camera_error"] == 0.0


def test_evaluate_samples_length_mismatch() -> None:
    with pytest.raises(ValueError, match="样本数不一致"):
        evaluate_samples([NormalizedAction.neutral()], [])
