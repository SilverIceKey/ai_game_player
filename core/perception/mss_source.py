"""WindowFrameSource：窗口定位 + mss 截屏。

平台约束（计划文档第 4 节）：mss 延迟导入（grab 内 import），
Linux 开发机不安装 mss 也可跑全部单元测试；Windows 实机才触达真实截屏。
窗口定位使用 Win32 API（ctypes 标准库，无新依赖）；非 Windows 平台给出明确报错。
"""
from __future__ import annotations

import sys

import numpy as np


def match_window_title(query: str, titles: list[str]) -> str | None:
    """在窗口标题列表中查找匹配项：先精确（忽略大小写），再包含匹配。

    纯函数，平台无关，供单元测试覆盖匹配策略。
    """
    exact = query.casefold()
    for title in titles:
        if title.casefold() == exact:
            return title
    for title in titles:
        if exact in title.casefold():
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
        windows = list_visible_windows()
        matched = match_window_title(self.window_title, [t for t, _ in windows])
        if matched is None:
            listing = "\n".join(f"  {t!r} {r}" for t, r in windows) or "  （无可见窗口）"
            raise RuntimeError(
                f"未找到窗口: {self.window_title!r}。当前可见窗口：\n{listing}\n"
                "请将 configs/wukong.yaml 的 window.title 改为上述标题之一"
                "（也可用 --list-windows 单独查看窗口列表）"
            )
        return dict(windows)[matched]
