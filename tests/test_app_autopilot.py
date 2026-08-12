"""app/autopilot.py 测试（spec §5.2 AUTOPILOT 安全链路 + §26/§27 correction 记录）。

全部用假 source / NullExecutor / PlaceholderPolicy / 假 InputCapture /
注入式 SafetyFilter（假按键轮询 + 假焦点检查）+ tmp_path writer，
不触达 win32 / pynput / pydirectinput / 真实截屏。
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from app.autopilot import AutopilotSession, parse_args
from capture.action import SOURCE_AI, SOURCE_CORRECTION, ActionRecord, NormalizedAction
from capture.clock import now_us
from capture.input.base import QueuedInputCapture
from config import CaptureConfig, GameConfig, ModelConfig, Settings, WindowConfig
from dataset.episode_store import EpisodeStoreReader, EpisodeStoreWriter
from model.policy import PlaceholderPolicy
from runtime.null_executor import NullExecutor
from runtime.safety_filter import MODE_HUMAN_OVERRIDE, SafetyFilter


class FakeSource:
    def __init__(self, width: int = 64, height: int = 36):
        self._frame = np.zeros((height, width, 3), dtype=np.uint8)

    def grab(self) -> tuple[np.ndarray, int]:
        return self._frame, now_us()


class FakeInputCapture(QueuedInputCapture):
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def emit(self, action: NormalizedAction) -> None:
        self._emit(ActionRecord(now_us(), action))


class FakePoller:
    def __init__(self):
        self.down = False

    def __call__(self, key: str) -> bool:
        return self.down


def _settings() -> Settings:
    return Settings(
        game="test",
        capture=CaptureConfig(source_fps=30.0),
        model=ModelConfig(history_frames=2, history_actions=2, input_width=64, input_height=36),
    )


def _game_config() -> GameConfig:
    return GameConfig(
        name="test",
        window=WindowConfig(title="Test Game"),
        keys={"move_forward": "w", "attack_light": "mouse_left"},
    )


def _make_session(tmp_path: Path):
    game_config = _game_config()
    executor = NullExecutor()
    poller = FakePoller()
    import runtime.action_scheduler as sched_mod

    scheduler = sched_mod.ActionScheduler(_settings().prediction.action_step_ms)
    safety = SafetyFilter(
        game_config.safety,
        window_title="test",
        on_release=executor.release_all,
        on_clear=scheduler.clear,
        key_poller=poller,
        focus_checker=lambda: True,
    )
    session = AutopilotSession(
        _settings(),
        game_config,
        source=FakeSource(),
        executor=executor,
        policy=PlaceholderPolicy(),
        input_capture=FakeInputCapture(),
        writer=EpisodeStoreWriter(
            tmp_path / "session",
            mode="AUTOPILOT",
            game="test",
            capture_width=64,
            capture_height=36,
            capture_fps=30.0,
            input_device="keyboard_mouse",
            dataset_version="dataset-v001",
        ),
        safety=safety,
        scheduler=scheduler,
    )
    return session, executor, poller


def _wait_for(cond, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_parse_args_requires_game() -> None:
    import pytest

    with pytest.raises(SystemExit):
        parse_args([])


def test_ai_loop_executes_and_records(tmp_path: Path) -> None:
    """AI 闭环：PlaceholderPolicy chunk → 调度 → NullExecutor；动作以 source=ai 记录。"""
    session, executor, _poller = _make_session(tmp_path)
    session.start()
    assert _wait_for(lambda: len(executor.actions) > 0)
    session.stop()

    reader = EpisodeStoreReader(tmp_path / "session")
    assert any(a.source == SOURCE_AI for a in reader.actions())
    assert len(reader.episodes()) == 1


def test_override_blocks_ai_and_records_correction(tmp_path: Path) -> None:
    """spec §26：F12 接管 → AI 动作被阻断；接管期玩家操作以 source=correction 记录。"""
    session, executor, poller = _make_session(tmp_path)
    session.start()
    assert _wait_for(lambda: len(executor.actions) > 0)

    poller.down = True  # 按下 override → HUMAN_OVERRIDE
    assert _wait_for(lambda: session._safety.mode == MODE_HUMAN_OVERRIDE)
    poller.down = False
    blocked_at = len(executor.actions)

    session._input_capture.emit(NormalizedAction(buttons=frozenset({"attack_light"})))
    time.sleep(0.3)  # 给 correction 线程写入窗口
    session.stop()

    assert len(executor.actions) == blocked_at  # 接管期间 AI 未再发动作
    corrections = [a for a in EpisodeStoreReader(tmp_path / "session").actions()
                   if a.source == SOURCE_CORRECTION]
    assert any(a.action.pressed("attack_light") for a in corrections)
