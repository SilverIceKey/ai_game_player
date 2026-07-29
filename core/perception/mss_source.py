"""WindowFrameSource：窗口定位 + mss 截屏。

平台约束（计划文档第 4 节）：mss 延迟导入（grab 内 import），
Linux 开发机不安装 mss 也可跑全部单元测试；Windows 实机才触达真实截屏。
窗口定位使用 Win32 API（ctypes 标准库，无新依赖）；非 Windows 平台给出明确报错。
"""
from __future__ import annotations

import sys

import numpy as np


class WindowFrameSource:
    """实现 core.perception.base.FrameSource 契约：锁定游戏窗口截屏。"""

    def __init__(self, window_title: str, rect: tuple[int, int, int, int] | None = None):
        self.window_title = window_title
        # rect=(x, y, w, h)：配置手动指定时跳过窗口定位（校准/调试用途）
        self._rect = rect
        self._sct = None

    def grab(self) -> np.ndarray:
        """抓取一帧 BGR 图像，形状 (H, W, 3)。"""
        import mss  # 延迟导入：开发机不安装 mss

        if self._sct is None:
            self._sct = mss.mss()
        if self._rect is None:
            self._rect = self._locate_window()
        x, y, w, h = self._rect
        shot = self._sct.grab({"left": x, "top": y, "width": w, "height": h})
        return np.asarray(shot)[:, :, :3].copy()  # BGRA → BGR

    def _locate_window(self) -> tuple[int, int, int, int]:
        if sys.platform != "win32":
            raise RuntimeError(
                "窗口定位仅支持 Windows 实机；开发机请跑单元测试，或在配置中手动指定 window.rect"
            )
        import ctypes
        from ctypes import wintypes

        hwnd = ctypes.windll.user32.FindWindowW(None, self.window_title)
        if not hwnd:
            raise RuntimeError(
                f"未找到窗口: {self.window_title!r}（请确认游戏已启动且窗口标题与配置一致）"
            )
        rect = wintypes.RECT()
        if not ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            raise RuntimeError(f"读取窗口区域失败: {self.window_title!r}")
        return (rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top)
