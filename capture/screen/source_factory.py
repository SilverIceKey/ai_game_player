"""截屏后端选择（spec §13 采集入口；迁移自旧 core/perception/source_factory.py）。

- mss：WindowFrameSource（按屏幕区域截屏，要求窗口在前台不被遮挡）
- wgc：WgcFrameSource（Windows.Graphics.Capture 窗口内容截屏，支持后台/遮挡）
- auto：优先 WGC，初始化或首帧握手失败时打日志降级 mss

两种后端的 grab() 统一返回 (frame_bgr, timestamp_us)（spec §11 统一时钟）。
"""
from __future__ import annotations

import logging

import config
from capture.screen.mss_source import WindowFrameSource


def _build_wgc(window_config: config.WindowConfig):
    """构建 WGC 源并做首帧握手：抓不到立刻暴露，auto 模式可及时降级。"""
    from capture.screen.wgc_source import WgcFrameSource

    source = WgcFrameSource(window_config.title)
    source.start()
    return source


def build_frame_source(
    window_config: config.WindowConfig, logger: logging.Logger | None = None
):
    """按 window_config.capture_backend 构建帧源。"""
    backend = window_config.capture_backend

    if backend == "mss":
        return WindowFrameSource(window_config.title, rect=window_config.rect)

    if backend == "wgc":
        # 强制 WGC：失败直接抛明确 RuntimeError，由 CLI 转用户可读报错
        return _build_wgc(window_config)

    # auto：WGC 优先，失败降级 mss
    log = logger or logging.getLogger("ai_game_player")
    try:
        return _build_wgc(window_config)
    except RuntimeError as exc:
        log.info("capture backend: WGC 不可用（%s），降级 mss 屏幕区域截屏", exc)
        return WindowFrameSource(window_config.title, rect=window_config.rect)
