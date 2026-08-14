"""runtime/voice_announcer.py 测试（PLAN-20260814-autopilot-voice-v1）。

FakeTTSClient 不触网；AutopilotSession 集成复用假组件模式（同 test_app_autopilot.py），
验证启动/接管/恢复/退出事件播报与决策节流。
"""
from __future__ import annotations

import time
from pathlib import Path

from app.autopilot import AutopilotSession
from capture.action import NormalizedAction
from config import CaptureConfig, GameConfig, ModelConfig, SafetyConfig, Settings, WindowConfig
from dataset.episode_store import EpisodeStoreWriter
from model.policy import PlaceholderPolicy
from runtime.null_executor import NullExecutor
from runtime.safety_filter import MODE_HUMAN_OVERRIDE, SafetyFilter
from runtime.voice_announcer import VoiceAnnouncer, format_action
from tests.test_app_autopilot import FakeInputCapture, FakeSource, _wait_for


class _FakeThread:
    def join(self, timeout=None) -> None:
        pass


class FakeTTSClient:
    """与 runtime.tts_client.TTSClient 同签名，记录播报文本，不触网。"""

    def __init__(self):
        self.spoken: list[str] = []

    def speak(self, text, speed=1.0, language=None, speaker=None, block=False):
        self.spoken.append(text)
        return _FakeThread()

    def stop(self) -> None:
        pass


# ---------- format_action 文案 ----------


def test_format_action_neutral_returns_none() -> None:
    assert format_action(NormalizedAction.neutral()) is None


def test_format_action_move_and_buttons() -> None:
    action = NormalizedAction(move_y=1.0, buttons=frozenset({"attack_light", "dodge"}))
    assert format_action(action) == "前进，轻击，闪避"


def test_format_action_deadzone_and_direction() -> None:
    assert format_action(NormalizedAction(move_y=0.1)) is None  # 死区内不播
    assert format_action(NormalizedAction(move_x=-0.5)) == "左移"
    assert format_action(NormalizedAction(move_x=0.5, move_y=-0.5)) == "后退，右移"


# ---------- 决策节流 ----------


def test_decision_throttle() -> None:
    client = FakeTTSClient()
    announcer = VoiceAnnouncer(client, decision_interval_s=0.2)
    announcer.speak_decision("前进")
    announcer.speak_decision("前进")  # 间隔内丢弃
    assert client.spoken == ["前进"]
    time.sleep(0.25)
    announcer.speak_decision("后退")
    assert client.spoken == ["前进", "后退"]


def test_decision_disabled_and_empty() -> None:
    client = FakeTTSClient()
    announcer = VoiceAnnouncer(client, decision_interval_s=0.0)  # 0 = 关闭
    announcer.speak_decision("前进")
    announcer.speak("事件")  # 事件直通不受 interval 影响
    assert client.spoken == ["事件"]
    announcer2 = VoiceAnnouncer(client, decision_interval_s=1.0)
    announcer2.speak_decision(None)  # 空文本丢弃
    announcer2.speak_decision("")
    assert client.spoken == ["事件"]


# ---------- AutopilotSession 事件播报集成 ----------


def _make_voice_session(tmp_path: Path):
    """resume_idle_ms=300 + announcer 的会话：接管/恢复可在测试时长内发生。"""
    import runtime.action_scheduler as sched_mod

    game_config = GameConfig(
        name="test",
        window=WindowConfig(title="Test Game"),
        keys={"move_forward": "w", "attack_light": "mouse_left"},
        safety=SafetyConfig(resume_idle_ms=300.0, max_action_rate_hz=1000.0),
    )
    executor = NullExecutor()
    scheduler = sched_mod.ActionScheduler(50.0)
    safety = SafetyFilter(
        game_config.safety,
        window_title="test",
        on_release=executor.release_all,
        on_clear=scheduler.clear,
        key_poller=lambda key: False,
        focus_checker=lambda: True,
    )
    client = FakeTTSClient()
    session = AutopilotSession(
        Settings(
            game="test",
            capture=CaptureConfig(source_fps=30.0),
            model=ModelConfig(history_frames=2, history_actions=2, input_width=64, input_height=36),
        ),
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
        announcer=VoiceAnnouncer(client, decision_interval_s=0.01),
    )
    return session, client


def test_autopilot_voice_events(tmp_path: Path) -> None:
    """启动/接管/恢复/退出各播报一次；接管期 AI 动作不下发也不播决策。"""
    session, client = _make_voice_session(tmp_path)
    session.start()
    assert _wait_for(lambda: "自动驾驶已启动" in client.spoken)
    assert _wait_for(lambda: len(session._action_history._records) > 0)  # AI 正常下发

    session._input_capture.emit(NormalizedAction(buttons=frozenset({"attack_light"})))
    assert _wait_for(lambda: session._safety.mode == MODE_HUMAN_OVERRIDE)  # 自动接管
    assert _wait_for(lambda: "已接管，交给你了" in client.spoken)

    assert _wait_for(
        lambda: session._safety.mode != MODE_HUMAN_OVERRIDE, timeout=3.0
    )  # 静默 300ms → 自动恢复
    assert _wait_for(lambda: "恢复 AI 控制" in client.spoken)

    session.stop()
    assert "自动驾驶已退出" in client.spoken
    # 事件各恰好一次
    assert client.spoken.count("自动驾驶已启动") == 1
    assert client.spoken.count("已接管，交给你了") == 1
    assert client.spoken.count("恢复 AI 控制") == 1
    assert client.spoken.count("自动驾驶已退出") == 1


class _MovingPolicy(PlaceholderPolicy):
    """恒输出"前进"动作：用于触发决策播报（PlaceholderPolicy 全 neutral 不会播）。"""

    def predict(self, frames, action_history, audio_pcm=None):
        from capture.action import ActionChunk

        return ActionChunk(
            actions=(NormalizedAction(move_y=1.0), NormalizedAction(move_y=1.0)),
            step_ms=50.0,
            model_version="moving-fake",
            confidence={"neutral": 1.0},
        )


def test_autopilot_voice_decision_announced(tmp_path: Path) -> None:
    """AI_CONTROL 下执行非空动作触发节流决策播报，文案来自 format_action。"""
    session, client = _make_voice_session(tmp_path)
    session._worker._policy = _MovingPolicy()
    session.start()
    assert _wait_for(lambda: "前进" in client.spoken)
    session.stop()
    decisions = [s for s in client.spoken
                 if s not in ("自动驾驶已启动", "自动驾驶已退出")]
    assert decisions
    assert all(format_action(NormalizedAction(move_y=1.0)) == s for s in decisions)
