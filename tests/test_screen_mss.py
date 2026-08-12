"""capture/screen/mss_source.py 单元测试。

纯逻辑（match_window_title）直接测；grab 路径通过向 sys.modules 注入
假 mss 模块覆盖（不安装 mss、不触达真实截屏）；窗口定位用
monkeypatch 替换 list_visible_windows。win32 路径（list_visible_windows
真实枚举）在 Linux 上只验证明确报错。
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from capture.clock import now_us
from capture.screen import mss_source
from capture.screen.mss_source import WindowFrameSource, match_window_title


# ---------- match_window_title 纯函数 ----------


def test_match_exact_case_insensitive():
    assert match_window_title("elden ring", ["ELDEN RING", "other"]) == "ELDEN RING"


def test_match_strips_whitespace():
    assert match_window_title("game  ", ["  Game", "other"]) == "  Game"


def test_match_falls_back_to_contains():
    titles = ["Black Myth: Wukong - 存档界面", "VSCode"]
    assert match_window_title("black myth", titles) == "Black Myth: Wukong - 存档界面"


def test_match_exact_beats_contains():
    titles = ["game 2", "game"]
    assert match_window_title("GAME", titles) == "game"


def test_match_none_when_not_found():
    assert match_window_title("不存在", ["a", "b"]) is None


# ---------- 假 mss 注入的 grab 路径 ----------


class _FakeShot:
    """模拟 mss ScreenShot：np.asarray 后得到 BGRA 数组。"""

    def __init__(self, bgra: np.ndarray):
        self._bgra = bgra

    def __array__(self, dtype=None, copy=None):
        arr = self._bgra.astype(dtype) if dtype else self._bgra
        return arr.copy() if copy else arr


class _FakeMss:
    def __init__(self, bgra: np.ndarray):
        self._bgra = bgra
        self.last_monitor = None

    def grab(self, monitor):
        self.last_monitor = monitor
        return _FakeShot(self._bgra)


@pytest.fixture
def fake_mss(monkeypatch):
    bgra = np.arange(4 * 6 * 4, dtype=np.uint8).reshape(4, 6, 4)
    fake = _FakeMss(bgra)
    module = types.ModuleType("mss")
    module.mss = lambda: fake
    monkeypatch.setitem(sys.modules, "mss", module)
    return fake, bgra


def test_grab_returns_bgr_frame_and_timestamp(fake_mss):
    fake, bgra = fake_mss
    source = WindowFrameSource("任意标题", rect=(10, 20, 6, 4))
    before = now_us()
    frame, timestamp_us = source.grab()
    after = now_us()

    assert frame.shape == (4, 6, 3)  # BGRA → BGR 去 alpha
    np.testing.assert_array_equal(frame, bgra[:, :, :3])
    assert isinstance(timestamp_us, int)
    assert before <= timestamp_us <= after  # 时间戳在抓帧完成时刻
    assert fake.last_monitor == {"left": 10, "top": 20, "width": 6, "height": 4}


def test_grab_locates_window_when_no_rect(fake_mss, monkeypatch):
    windows = [("我的游戏窗口", (100, 200, 1280, 720)), ("别的", (0, 0, 10, 10))]
    monkeypatch.setattr(mss_source, "list_visible_windows", lambda: windows)
    fake, _ = fake_mss
    source = WindowFrameSource("游戏窗口")
    source.grab()
    assert fake.last_monitor == {"left": 100, "top": 200, "width": 1280, "height": 720}


def test_locate_window_error_lists_visible_windows(fake_mss, monkeypatch):
    monkeypatch.setattr(
        mss_source, "list_visible_windows", lambda: [("窗口甲", (0, 0, 1, 1))]
    )
    source = WindowFrameSource("找不到的标题")
    with pytest.raises(RuntimeError, match="窗口甲"):
        source.grab()


def test_list_visible_windows_requires_windows():
    if sys.platform == "win32":
        pytest.skip("仅非 Windows 平台验证报错路径")
    with pytest.raises(RuntimeError, match="仅支持 Windows"):
        mss_source.list_visible_windows()
