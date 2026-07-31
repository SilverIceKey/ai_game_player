"""M3 仲裁器测试：优先级、动作有效期、急停阻断、仲裁日志。"""
import logging

import pytest

from core.contracts import Action
from core.control.arbiter import (
    ActionArbiter,
    ActionSource,
    CandidateAction,
)


def _arb(now=10.0, **kwargs) -> ActionArbiter:
    kwargs.setdefault("logger", logging.getLogger("test-arbiter-null"))
    return ActionArbiter(clock=lambda: now, **kwargs)


def _cand(name, source, issued_at, ttl=None):
    return CandidateAction(Action(name), source, issued_at, ttl)


def test_priority_reflex_over_skill():
    arb = _arb()
    decision = arb.decide([
        _cand("move", ActionSource.SKILL, 9.9),
        _cand("dodge", ActionSource.REFLEX, 9.8),
    ])
    assert decision.action.name == "dodge"
    assert decision.winner == ActionSource.REFLEX
    assert (ActionSource.SKILL, "preempted") in decision.dropped


def test_priority_order_constant():
    assert ActionSource.EMERGENCY > ActionSource.FOCUS_GUARD > ActionSource.REFLEX > ActionSource.SKILL


def test_expired_action_dropped():
    arb = _arb(now=10.0)  # 默认 ttl 500ms
    decision = arb.decide([_cand("move", ActionSource.SKILL, 9.0)])  # 已过期 1000ms
    assert decision.action is None
    assert decision.winner is None
    assert (ActionSource.SKILL, "expired") in decision.dropped


def test_fresh_action_executed():
    arb = _arb(now=10.0)
    decision = arb.decide([_cand("move", ActionSource.SKILL, 9.8)])  # 200ms，未过期
    assert decision.action.name == "move"
    assert decision.dropped == ()


def test_custom_ttl_from_arbiter_default():
    arb = _arb(now=10.0, default_ttl_ms=100.0)
    decision = arb.decide([_cand("move", ActionSource.SKILL, 9.85)])  # 150ms > 100ms
    assert decision.action is None
    # 候选自带 ttl 时覆盖仲裁器默认
    decision = arb.decide([_cand("move", ActionSource.SKILL, 9.85, ttl=500.0)])
    assert decision.action.name == "move"


def test_emergency_blocks_everything():
    arb = _arb()
    decision = arb.decide(
        [_cand("move", ActionSource.SKILL, 9.9), _cand("dodge", ActionSource.REFLEX, 9.9)],
        blocked=True,
        blocked_reason="人工急停中",
    )
    assert decision.action is None
    assert len(decision.dropped) == 2
    assert all(reason.startswith("blocked") for _, reason in decision.dropped)


def test_arbiter_logs_decision(caplog):
    logger = logging.getLogger("test-arbiter-log")
    arb = ActionArbiter(clock=lambda: 10.0, logger=logger)
    with caplog.at_level(logging.INFO, logger="test-arbiter-log"):
        arb.decide([
            _cand("move", ActionSource.SKILL, 9.9),
            _cand("dodge", ActionSource.REFLEX, 9.8),
        ])
    text = caplog.text
    assert "arbiter winner=reflex action=dodge" in text
    assert "skill:preempted" in text
