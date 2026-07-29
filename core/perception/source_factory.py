"""截屏后端选择（实机联调需求：游戏窗口被 VSCode 遮挡时需后台截屏）。

- mss：WindowFrameSource（按屏幕区域截屏，要求窗口在前台不被遮挡）
- wgc：WgcFrameSource（Windows.Graphics.Capture 窗口内容截屏，支持后台/遮挡）
- auto：优先 WGC，初始化或首帧握手失败时打日志降级 mss

build_wukong 与 run_calibrate_cli 共用本工厂，calibrate 同样支持后台截屏。
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from core.perception.mss_source import WindowFrameSource

if TYPE_CHECKING:
    from games.wukong.adapter import WindowConfig


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _build_wgc(window_config: WindowConfig):
    """构建 WGC 源并做首帧握手：抓不到立刻暴露，auto 模式可及时降级。"""
    from core.perception.wgc_source import WgcFrameSource

    source = WgcFrameSource(window_config.title)
    source.start()
    return source


def build_frame_source(window_config: WindowConfig, logger: logging.Logger | None = None):
    """按 window_config.capture_backend 构建 FrameSource。"""
    backend = window_config.capture_backend

    if backend == "mss":
        return WindowFrameSource(window_config.title, rect=window_config.rect)

    if backend == "wgc":
        # 强制 WGC：失败直接抛明确 RuntimeError，由 CLI 转用户可读报错
        return _build_wgc(window_config)

    # auto：WGC 优先，失败降级 mss
    log = logger or logging.getLogger("auto_player")
    try:
        return _build_wgc(window_config)
    except RuntimeError as exc:
        log.info(
            "[%s] capture backend: WGC 不可用（%s），降级 mss 屏幕区域截屏", _ts(), exc
        )
        return WindowFrameSource(window_config.title, rect=window_config.rect)
