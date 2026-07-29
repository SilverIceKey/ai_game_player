"""WgcFrameSource：Windows.Graphics.Capture 后台截屏（windows-capture 包）。

与 WindowFrameSource（按屏幕区域截屏）不同，WGC 截的是窗口内容本身，
游戏窗口被 VSCode 等遮挡时仍能拿到游戏画面——VSCode 终端启动场景必需。

平台约束（同 mss 源的老规矩）：windows_capture 构造时延迟导入，
Linux 开发机不安装该包也可跑全部单元测试；非 Windows / 包缺失 /
包 API 与 v1.x 不符，均抛带包版本提示的明确 RuntimeError，
由 source_factory（auto 降级）或 CLI（SystemExit）转成用户可读报错。

包是异步回调式 API，这里用「最新一帧缓冲 + Condition」适配同步 grab() 契约。
"""
from __future__ import annotations

import sys
import threading

import numpy as np


class WgcFrameSource:
    """实现 core.perception.base.FrameSource 契约：窗口内容截屏（支持遮挡/后台）。"""

    def __init__(self, window_title: str, first_frame_timeout: float = 5.0):
        if sys.platform != "win32":
            raise RuntimeError(
                "WGC 后台截屏仅支持 Windows 实机（Windows.Graphics.Capture API）；"
                "开发机请用 capture_backend: mss"
            )
        try:
            import windows_capture  # 延迟导入：Linux 开发机不安装
        except ImportError as exc:
            raise RuntimeError(
                "未安装 windows-capture（pyproject 已按 sys_platform=='win32' 声明）；"
                "Windows 实机请执行: pip install \"windows-capture>=1.5,<2\""
            ) from exc

        self._wc_version = getattr(windows_capture, "__version__", "未知")
        capture_cls = getattr(windows_capture, "WindowsCapture", None)
        if capture_cls is None:
            raise RuntimeError(self._api_error("缺少 WindowsCapture 类"))
        try:
            self._capture = capture_cls(
                cursor_capture=None,
                draw_border=None,
                monitor_index=None,
                window_name=window_title,
            )
        except TypeError as exc:
            raise RuntimeError(self._api_error(f"构造参数不符: {exc}")) from exc

        self.window_title = window_title
        self.first_frame_timeout = float(first_frame_timeout)
        self._cond = threading.Condition()
        self._latest: np.ndarray | None = None
        self._closed = False
        self._started = False

        try:
            @self._capture.event
            def on_frame_arrived(frame, capture_control):  # noqa: ARG004 包回调签名
                buf = getattr(frame, "frame_buffer", None)
                # 防御 API 变化：非 BGRA 缓冲不更新，保留上一帧
                if buf is None or buf.ndim != 3 or buf.shape[2] != 4:
                    return
                with self._cond:
                    self._latest = buf[:, :, :3].copy()  # BGRA → BGR
                    self._cond.notify_all()

            # windows-capture 强制要求 on_closed 处理器，缺失时 start() 报
            # "on_closed Event Handler Is Not Set"（v1.x 实机验证）
            @self._capture.event
            def on_closed():
                with self._cond:
                    self._closed = True
                    self._cond.notify_all()
        except (AttributeError, TypeError) as exc:
            raise RuntimeError(self._api_error(f"event 回调注册不符: {exc}")) from exc

    def _api_error(self, detail: str) -> str:
        return (
            f"windows-capture API 与预期（v1.x）不符: {detail}"
            f"（当前包版本: {self._wc_version}）。请核对已安装版本: pip show windows-capture"
        )

    def start(self) -> None:
        """启动捕获并阻塞等待首帧（带超时）。幂等；超时抛明确错误。"""
        if self._started:
            return
        try:
            self._capture.start()
        except TypeError as exc:
            raise RuntimeError(self._api_error(f"start() 调用不符: {exc}")) from exc
        with self._cond:
            if self._latest is None and not self._closed:
                arrived = self._cond.wait_for(
                    lambda: self._latest is not None or self._closed,
                    timeout=self.first_frame_timeout,
                )
                if not arrived:
                    raise RuntimeError(
                        f"WGC 首帧超时（{self.first_frame_timeout:g}s）: "
                        f"窗口 {self.window_title!r} 未抓到画面"
                        "（窗口不存在/被最小化，或包 API 不符；"
                        "可用 --list-windows 核对窗口标题）"
                    )
            if self._closed and self._latest is None:
                raise RuntimeError(
                    f"WGC 捕获在首帧前被关闭: 窗口 {self.window_title!r}"
                    "（窗口可能已关闭，可用 --list-windows 核对）"
                )
        self._started = True

    def grab(self) -> np.ndarray:
        """返回最近一帧 BGR 图像，形状 (H, W, 3)；首帧前阻塞等待（带超时）。"""
        self.start()
        with self._cond:
            if self._closed:
                raise RuntimeError(f"WGC 捕获已关闭: 窗口 {self.window_title!r}（窗口已关闭？）")
            latest = self._latest
        assert latest is not None  # start() 已保证
        return latest.copy()
