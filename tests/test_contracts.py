"""契约层冒烟测试：协议可导入、数据结构可实例化、实现满足 Protocol。"""
import time

import numpy as np

from core.contracts import Action, GameState, Suggestion
from core.control.base import Controller, Result
from core.decision.base import DecisionEngine
from core.perception.base import FrameSource
from core.recorder.base import Recorder, StepRecord
from games.base import GameAdapter
from llm.base import LLMProvider, ReviewEngine, ReviewReport


def test_contract_dataclasses():
    state = GameState(timestamp=time.time(), scene="combat")
    action = Action(name="light_attack")
    suggestion = Suggestion(action=action, reason="test", confidence=0.5)
    record = StepRecord(timestamp=state.timestamp, state=state, output=action)
    report = ReviewReport(summary="ok")

    assert suggestion.confidence == 0.5
    assert record.output is action
    assert report.issues == []


class _DummyAdapter:
    def perceive(self, frame: np.ndarray) -> GameState:
        return GameState(timestamp=0.0)

    def available_actions(self, state: GameState) -> list[Action]:
        return []

    def action_space(self) -> list[str]:
        return []


class _DummyController:
    def execute(self, action: Action) -> Result:
        return Result(success=True)


def test_implementations_satisfy_protocols():
    assert isinstance(_DummyAdapter(), GameAdapter)
    assert isinstance(_DummyController(), Controller)
