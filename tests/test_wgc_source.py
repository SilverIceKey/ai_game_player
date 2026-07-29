"""WgcFrameSource 事件注册与关闭路径测试（伪造 windows_capture 包）。

实机教训（v1.x）：包强制要求同时注册 on_frame_arrived 与 on_closed，
缺 on_closed 时 start() 报 "on_closed Event Handler Is Not Set"。
"""
import sys
import threading
import types

import numpy as np
import pytest

from core.perception.wgc_source import WgcFrameSource


class _FakeFrame:
    def __init__(self, buf):
        self.frame_buffer = buf


class _FakeCapture:
    """模拟 windows-capture v1.x：事件装饰器 + 强制双处理器校验。"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.handlers = {}
        self.push_mode = "frame"  # frame / closed / nothing / block / raise

    def event(self, fn):
        self.handlers[fn.__name__] = fn
        return fn

    def start(self):
        if "on_frame_arrived" not in self.handlers:
            raise Exception("on_frame_arrived Event Handler Is Not Set")
        if "on_closed" not in self.handlers:
            raise Exception("on_closed Event Handler Is Not Set")
        if self.push_mode == "frame":
            buf = np.zeros((4, 6, 4), dtype=np.uint8)
            self.handlers["on_frame_arrived"](_FakeFrame(buf), None)
        elif self.push_mode == "closed":
            self.handlers["on_closed"]()
        elif self.push_mode == "block":
            # 实机行为：start() 阻塞（在调用线程跑捕获循环）
            threading.Event().wait()
        elif self.push_mode == "raise":
            raise RuntimeError("模拟 start() 内部抛错")


@pytest.fixture
def fake_wc(monkeypatch):
    """注入伪造的 windows_capture 包并伪装为 win32。返回最近构造的 capture 引用。"""
    holder = {}

    def factory(**kwargs):
        holder["capture"] = _FakeCapture(**kwargs)
        return holder["capture"]

    module = types.ModuleType("windows_capture")
    module.WindowsCapture = factory
    module.__version__ = "fake"
    monkeypatch.setitem(sys.modules, "windows_capture", module)
    monkeypatch.setattr(sys, "platform", "win32")
    return holder


def test_registers_both_handlers_and_grabs_frame(fake_wc):
    source = WgcFrameSource("b1")
    capture = fake_wc["capture"]
    assert set(capture.handlers) >= {"on_frame_arrived", "on_closed"}

    frame = source.grab()
    assert frame.shape == (4, 6, 3)  # BGRA → BGR


def test_grab_raises_when_closed_before_first_frame(fake_wc):
    source = WgcFrameSource("b1")
    fake_wc["capture"].push_mode = "closed"
    with pytest.raises(RuntimeError, match="关闭"):
        source.grab()


def test_grab_raises_after_capture_closed(fake_wc):
    source = WgcFrameSource("b1")
    source.grab()  # 首帧成功
    fake_wc["capture"].handlers["on_closed"]()
    with pytest.raises(RuntimeError, match="已关闭"):
        source.grab()


def test_first_frame_timeout(fake_wc):
    source = WgcFrameSource("b1", first_frame_timeout=0.05)
    fake_wc["capture"].push_mode = "nothing"
    with pytest.raises(RuntimeError, match="首帧超时"):
        source.grab()


def test_blocking_start_does_not_freeze_caller(fake_wc):
    """start() 阻塞（实机行为）时调用方不卡死：守护线程承载，走首帧超时路径。"""
    source = WgcFrameSource("b1", first_frame_timeout=0.05)
    fake_wc["capture"].push_mode = "block"
    with pytest.raises(RuntimeError, match="首帧超时"):
        source.grab()


def test_start_internal_error_propagates(fake_wc):
    """start() 在守护线程内抛错时，主线程拿到明确 RuntimeError 而不是挂起。"""
    source = WgcFrameSource("b1", first_frame_timeout=5.0)
    fake_wc["capture"].push_mode = "raise"
    with pytest.raises(RuntimeError, match="内部抛错"):
        source.grab()
