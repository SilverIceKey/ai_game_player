"""把游戏窗口提前台（ctypes Win32，不引入 pywin32）。

正式跑（非 dry-run）时 mss 后端要求游戏窗口在前台不被遮挡。
SetForegroundWindow 常被系统焦点拦截（进程不持有前台权限时调用被忽略），
因此用 AttachThreadInput 挂靠到当前前台线程做兜底（Windows 通用技巧）。
"""
from __future__ import annotations

import sys

from core.perception.mss_source import match_window_title

_SW_RESTORE = 9


def list_window_handles() -> list[tuple[str, int]]:
    """枚举当前可见顶层窗口：[(标题, hwnd)]。非 Windows 返回空列表。"""
    if sys.platform != "win32":
        return []
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    results: list[tuple[str, int]] = []
    WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def _callback(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        results.append((buf.value, hwnd))
        return True

    user32.EnumWindows(WNDENUMPROC(_callback), 0)
    return results


def bring_to_foreground(window_title: str) -> bool:
    """把标题匹配的窗口提前台（先精确后包含匹配，同截屏窗口定位）。

    返回是否成功置为前台；非 Windows / 未找到窗口 / 被系统拦截返回 False。
    """
    if sys.platform != "win32":
        return False
    import ctypes

    windows = list_window_handles()
    matched = match_window_title(window_title, [title for title, _ in windows])
    if matched is None:
        return False
    hwnd = dict(windows)[matched]

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    user32.ShowWindow(hwnd, _SW_RESTORE)

    current_tid = kernel32.GetCurrentThreadId()
    foreground_tid = user32.GetWindowThreadProcessId(user32.GetForegroundWindow(), None)
    attached = False
    if foreground_tid and foreground_tid != current_tid:
        attached = bool(user32.AttachThreadInput(current_tid, foreground_tid, True))
    try:
        user32.SetForegroundWindow(hwnd)
        user32.BringWindowToTop(hwnd)
    finally:
        if attached:
            user32.AttachThreadInput(current_tid, foreground_tid, False)
    return bool(user32.GetForegroundWindow() == hwnd)
