"""WindowFrameSource：窗口定位 + mss 截屏（spec §11：抓帧时刻打统一时钟）。

迁移自旧 core/perception/mss_source.py。改造点（重构计划 §2.1）：
`grab()` 返回 `(frame_bgr_ndarray, timestamp_us)`，时间戳在 mss.grab 完成
的时刻用 `capture.clock.now_us()` 打点——spec §11 要求"某一帧对应玩家
哪一个操作"可精确回答，因此时间戳必须贴近真实抓帧时刻。

平台约束：mss 延迟导入（grab 内 import），Linux 开发机不安装 mss 也可跑
全部单元测试；窗口定位用 Win32 API（ctypes 标准库），非 Windows 明确报错。
"""
from __future__ import annotations

import sys

import numpy as np

from capture.clock import now_us


def match_window_title(query: str, titles: list[str]) -> str | None:
    """在窗口标题列表中查找匹配项：先精确（忽略大小写与首尾空格），再包含匹配。

    纯函数，平台无关，供单元测试覆盖匹配策略。
    忽略首尾空格：配置手误（如 "b1  "）或标题带不可见空格时不致匹配失败。
    """
    exact = query.strip().casefold()
    for title in titles:
        if title.strip().casefold() == exact:
            return title
    for title in titles:
        if exact in title.strip().casefold():
            return title
    return None


def list_visible_windows() -> list[tuple[str, tuple[int, int, int, int]]]:
    """枚举当前可见顶层窗口：[(标题, (x, y, w, h))]。仅 Windows 实机可用。"""
    if sys.platform != "win32":
        raise RuntimeError("窗口枚举仅支持 Windows 实机；开发机无法列出窗口")
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    results: list[tuple[str, tuple[int, int, int, int]]] = []
    WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def _callback(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        rect = wintypes.RECT()
        if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            results.append(
                (buf.value, (rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top))
            )
        return True

    user32.EnumWindows(WNDENUMPROC(_callback), 0)
    return results


class WindowFrameSource:
    """屏幕区域截屏帧源：锁定游戏窗口矩形，逐帧 grab。"""

    def __init__(self, window_title: str, rect: tuple[int, int, int, int] | None = None):
        self.window_title = window_title
        # rect=(x, y, w, h)：配置手动指定时跳过窗口定位（校准/调试用途）
        self._rect = rect
        self._sct = None

    def grab(self) -> tuple[np.ndarray, int]:
        """抓取一帧，返回 (BGR 图像 (H, W, 3), 抓帧完成时刻 timestamp_us)。"""
        import mss  # 延迟导入：开发机不安装 mss

        if self._sct is None:
            self._sct = mss.mss()
        if self._rect is None:
            self._rect = self._locate_window()
        x, y, w, h = self._rect
        shot = self._sct.grab({"left": x, "top": y, "width": w, "height": h})
        timestamp_us = now_us()  # §11：在抓帧完成时刻打统一时钟
        return np.asarray(shot)[:, :, :3].copy(), timestamp_us  # BGRA → BGR

    def _locate_window(self) -> tuple[int, int, int, int]:
        windows = list_visible_windows()
        matched = match_window_title(self.window_title, [t for t, _ in windows])
        if matched is None:
            listing = "\n".join(f"  {t!r} {r}" for t, r in windows) or "  （无可见窗口）"
            raise RuntimeError(
                f"未找到窗口: {self.window_title!r}。当前可见窗口：\n{listing}\n"
                "请把游戏配置的 window.title 改为上述标题之一"
                "（也可用窗口枚举工具单独查看窗口列表）"
            )
        return dict(windows)[matched]
