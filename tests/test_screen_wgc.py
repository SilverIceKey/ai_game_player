"""capture/screen/wgc_source.py 与 foreground.py 单元测试。

WGC 在 Linux 上通过 monkeypatch sys.platform + 注入假 windows_capture
模块走通完整帧回调 → 缓冲 → grab 路径（不触达真实 WGC API）；
重点验证「每个缓冲帧携带自己的时间戳，grab 返回帧自身时间戳」。
foreground 的非 Windows 安全默认（空列表 / False）直接验证。
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from capture.screen import foreground
from capture.screen.wgc_source import WgcFrameSource


# ---------- 假 windows_capture 包 ----------


class _FakeFrame:
    def __init__(self, bgra: np.ndarray):
        self.frame_buffer = bgra


class _FakeWindowsCapture:
    """记录 event 注册的回调，start() 时按脚本触发帧事件。"""

    scripted_frames: list[np.ndarray] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._handlers = {}

    def event(self, func):
        self._handlers[func.__name__] = func
        return func

    def start(self):
        for bgra in self.scripted_frames:
            self._handlers["on_frame_arrived"](_FakeFrame(bgra), None)
        # 阻塞式 start：脚本播完就返回（守护线程结束即可）


@pytest.fixture
def fake_windows_capture(monkeypatch):
    module = types.ModuleType("windows_capture")
    module.__version__ = "1.5.0-fake"
    module.WindowsCapture = _FakeWindowsCapture
    monkeypatch.setitem(sys.modules, "windows_capture", module)
    monkeypatch.setattr(sys, "platform", "win32")
    yield module
    monkeypatch.undo()


# ---------- WgcFrameSource ----------


def test_wgc_raises_on_non_windows():
    if sys.platform == "win32":
        pytest.skip("仅非 Windows 平台验证报错路径")
    with pytest.raises(RuntimeError, match="仅支持 Windows"):
        WgcFrameSource("某窗口")


def test_wgc_grab_returns_frame_with_own_timestamp(fake_windows_capture):
    bgra1 = np.full((2, 3, 4), 10, dtype=np.uint8)
    bgra2 = np.full((2, 3, 4), 20, dtype=np.uint8)
    _FakeWindowsCapture.scripted_frames = [bgra1, bgra2]

    source = WgcFrameSource("游戏窗口")
    frame, ts1 = source.grab()
    np.testing.assert_array_equal(frame, bgra2[:, :, :3])  # 最新一帧
    assert isinstance(ts1, int) and ts1 > 0

    # 再注入一帧：grab 必须返回该帧自己的时间戳，且单调不减
    source._capture._handlers["on_frame_arrived"](_FakeFrame(bgra1), None)
    frame, ts2 = source.grab()
    np.testing.assert_array_equal(frame, bgra1[:, :, :3])
    assert ts2 >= ts1


def test_wgc_first_frame_timeout(fake_windows_capture):
    _FakeWindowsCapture.scripted_frames = []  # 永远不来帧
    source = WgcFrameSource("游戏窗口", first_frame_timeout=0.05)
    with pytest.raises(RuntimeError, match="首帧超时"):
        source.start()


def test_wgc_malformed_frame_keeps_previous(fake_windows_capture):
    bgra = np.full((2, 3, 4), 7, dtype=np.uint8)
    _FakeWindowsCapture.scripted_frames = [bgra]
    source = WgcFrameSource("游戏窗口")
    source.start()

    # 非 BGRA 的畸形帧不更新缓冲
    source._capture._handlers["on_frame_arrived"](_FakeFrame(np.zeros((2, 3, 3), np.uint8)), None)
    frame, _ = source.grab()
    np.testing.assert_array_equal(frame, bgra[:, :, :3])


# ---------- foreground ----------


def test_list_window_handles_non_windows_returns_empty():
    if sys.platform == "win32":
        pytest.skip("仅非 Windows 平台验证安全默认")
    assert foreground.list_window_handles() == []


def test_bring_to_foreground_non_windows_returns_false():
    if sys.platform == "win32":
        pytest.skip("仅非 Windows 平台验证安全默认")
    assert foreground.bring_to_foreground("任意窗口") is False
