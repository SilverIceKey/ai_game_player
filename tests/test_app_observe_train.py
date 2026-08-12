"""app/observe_train.py 测试（spec §5.1 OBSERVE_TRAIN + §41 SHADOW）。

全部用假 source / 假 InputCapture / 假热键轮询 + tmp_path writer，
不触达 win32 / pynput / 真实截屏。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

from app.common import build_input_capture, resolve_settings_path
from app.observe_train import (
    EpisodeHotkey,
    ObserveTrainSession,
    ShadowRunner,
    parse_args,
)
from capture.action import SOURCE_HUMAN, ActionRecord, NormalizedAction
from capture.clock import now_us
from capture.input.base import QueuedInputCapture
from capture.input.keyboard_mouse import KeyboardMouseCapture
from config import (
    CaptureConfig,
    GameConfig,
    ModelConfig,
    Settings,
    WindowConfig,
)
from dataset.episode_store import EpisodeStoreReader, EpisodeStoreWriter
from model.policy import PlaceholderPolicy


# ---------- 假组件 ----------


class FakeSource:
    """固定尺寸假帧源（grab 返回 (BGR 帧, timestamp_us)）。"""

    def __init__(self, width: int = 64, height: int = 36):
        self._frame = np.zeros((height, width, 3), dtype=np.uint8)

    def grab(self) -> tuple[np.ndarray, int]:
        return self._frame, now_us()


class FakeInputCapture(QueuedInputCapture):
    """内存队列假输入采集：emit 注入事件，绝不触达真实输入设备。"""

    def __init__(self):
        super().__init__()
        self.started = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def emit(self, action: NormalizedAction, source: str = SOURCE_HUMAN) -> None:
        self._emit(ActionRecord(now_us(), action, source))


class FakePoller:
    """可变按键状态的假热键轮询。"""

    def __init__(self):
        self.down = False

    def __call__(self, key: str) -> bool:
        return self.down


def _settings(**overrides) -> Settings:
    params = {
        "game": "test",
        "capture": CaptureConfig(source_fps=30.0),
        "model": ModelConfig(history_frames=4, input_width=64, input_height=36),
    }
    params.update(overrides)
    return Settings(**params)


def _game_config() -> GameConfig:
    return GameConfig(
        name="test",
        window=WindowConfig(title="Test Game"),
        keys={"move_forward": "w", "attack_light": "mouse_left"},
    )


def _writer(tmp_path: Path) -> EpisodeStoreWriter:
    return EpisodeStoreWriter(
        tmp_path / "session",
        mode="OBSERVE_TRAIN",
        game="test",
        capture_width=64,
        capture_height=36,
        capture_fps=30.0,
        input_device="keyboard_mouse",
        dataset_version="dataset-v001",
    )


def _wait_for(cond, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.02)
    return False


# ---------- CLI 参数解析 ----------


def test_parse_args_list_windows_no_game() -> None:
    args = parse_args(["--list-windows"])
    assert args.list_windows is True
    assert args.game is None


def test_parse_args_requires_game() -> None:
    with pytest.raises(SystemExit):
        parse_args([])


def test_parse_args_shadow_flag() -> None:
    args = parse_args(["--game", "wukong", "--shadow"])
    assert args.game == "wukong"
    assert args.shadow is True


# ---------- 配置路径回退（沿用旧 CLI 风格） ----------


def test_resolve_settings_path_fallback(tmp_path: Path) -> None:
    example = tmp_path / "settings.example.yaml"
    example.write_text("game: wukong\n", encoding="utf-8")
    resolved = resolve_settings_path(str(tmp_path / "settings.yaml"), "observe_train")
    assert resolved == example


def test_resolve_settings_path_missing(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="不存在"):
        resolve_settings_path(str(tmp_path / "settings.yaml"), "observe_train")


# ---------- 输入采集装配 ----------


def test_build_input_capture_keyboard_mouse() -> None:
    capture = build_input_capture(_settings(), _game_config())
    assert isinstance(capture, KeyboardMouseCapture)
    # 反推映射生效：按 w → move_forward → move_y=1.0 快照
    action = capture._mapper.on_key("w", True)
    assert action is not None
    assert action.move_y == 1.0


@pytest.mark.skipif(sys.platform == "win32", reason="仅非 Windows 验证 XInput 缺失报错")
def test_build_input_capture_gamepad_non_windows() -> None:
    with pytest.raises(RuntimeError, match="Windows"):
        build_input_capture(_settings(input_device="gamepad"), _game_config())


# ---------- Episode 热键 toggle ----------


def test_hotkey_toggle_edge_detection() -> None:
    events: list[str] = []
    poller = FakePoller()
    hotkey = EpisodeHotkey(
        "F9",
        on_start=lambda: events.append("start"),
        on_stop=lambda: events.append("stop"),
        key_poller=poller,
    )
    assert hotkey.supported is True

    hotkey.poll_once()  # 未按下：无事件
    assert events == []
    poller.down = True
    hotkey.poll_once()  # 按下沿：start
    hotkey.poll_once()  # 持续按下：不重复触发
    assert events == ["start"]
    poller.down = False
    hotkey.poll_once()
    poller.down = True
    hotkey.poll_once()  # 第二次按下沿：stop
    assert events == ["start", "stop"]


# ---------- 全链路 session 测试 ----------


def test_session_fallback_single_episode(tmp_path: Path) -> None:
    """无热键（hotkey=None，模拟非 Windows）：整 session 退化为单 episode。"""
    writer = _writer(tmp_path)
    source = FakeSource()
    input_capture = FakeInputCapture()
    session = ObserveTrainSession(
        _settings(),
        _game_config(),
        source=source,
        input_capture=input_capture,
        writer=writer,
        hotkey=None,
    )
    session.start()
    assert session.episode_active is True  # 自动开始单 episode
    input_capture.emit(NormalizedAction(move_y=1.0))
    time.sleep(0.5)
    session.stop()

    reader = EpisodeStoreReader(tmp_path / "session")
    episodes = reader.episodes()
    assert len(episodes) == 1
    assert episodes[0]["source"] == "human"
    assert len(reader.frames()) > 0
    actions = reader.actions()
    assert any(a.source == SOURCE_HUMAN and a.action.move_y == 1.0 for a in actions)
    assert reader.manifest()["mode"] == "OBSERVE_TRAIN"


def test_session_hotkey_episode_toggle(tmp_path: Path) -> None:
    """热键 toggle：F9 第一次按下开始 episode，第二次结束；帧只在 episode 内写入。"""
    writer = _writer(tmp_path)
    source = FakeSource()
    input_capture = FakeInputCapture()
    poller = FakePoller()
    session = ObserveTrainSession(
        _settings(),
        _game_config(),
        source=source,
        input_capture=input_capture,
        writer=writer,
        hotkey=None,
    )
    hotkey = EpisodeHotkey(
        "F9",
        on_start=session._begin_episode,
        on_stop=session._end_episode,
        key_poller=poller,
    )
    session.set_hotkey(hotkey)
    session.start()
    assert session.episode_active is False

    time.sleep(0.2)  # episode 未开始：抓了帧但不写
    assert session.episode_active is False

    poller.down = True  # 第一次按下 → START
    assert _wait_for(lambda: session.episode_active)
    poller.down = False
    input_capture.emit(NormalizedAction(buttons=frozenset({"attack_light"})))
    assert _wait_for(lambda: len(reader_frames(tmp_path)) > 0)

    poller.down = True  # 第二次按下 → STOP
    assert _wait_for(lambda: not session.episode_active)
    poller.down = False
    session.stop()

    reader = EpisodeStoreReader(tmp_path / "session")
    assert len(reader.episodes()) == 1
    assert any(
        a.action.pressed("attack_light") for a in reader.actions()
    )


def reader_frames(tmp_path: Path) -> list[dict]:
    idx = tmp_path / "session" / "frames.idx"
    if not idx.is_file():
        return []
    return [line for line in idx.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_session_shadow_metrics(tmp_path: Path) -> None:
    """SHADOW（spec §41）：PlaceholderPolicy 推理不执行，退出时累计对齐指标。"""
    writer = _writer(tmp_path)
    source = FakeSource()
    input_capture = FakeInputCapture()
    shadow = ShadowRunner(PlaceholderPolicy(), _settings())
    session = ObserveTrainSession(
        _settings(),
        _game_config(),
        source=source,
        input_capture=input_capture,
        writer=writer,
        hotkey=None,
        shadow=shadow,
    )
    session.start()
    input_capture.emit(NormalizedAction(move_y=1.0))
    # history_frames=4 @30fps：窗口迅速采满，等推理跑几轮
    assert _wait_for(lambda: shadow.metrics.count >= 2)
    session.stop()

    summary = shadow.metrics.summary()
    assert summary["sample_count"] >= 2
    # PlaceholderPolicy 恒输出 neutral：与玩家 move_y=1.0 的误差被计入
    assert "buttons" in summary
    rendered = shadow.render()
    assert "shadow metrics" in rendered
    assert "shadow_inference_ms" in rendered
