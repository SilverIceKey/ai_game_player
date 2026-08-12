"""capture/screen/source_factory.py 单元测试：后端选择与 auto 降级逻辑。

不触达真实 WGC/mss：_build_wgc 用 monkeypatch 替换。
"""
from __future__ import annotations

import logging

import pytest

import config
from capture.screen import source_factory
from capture.screen.mss_source import WindowFrameSource


def _window(backend: str) -> config.WindowConfig:
    return config.WindowConfig(title="游戏窗口", capture_backend=backend)


def test_backend_mss_returns_window_source():
    source = source_factory.build_frame_source(_window("mss"))
    assert isinstance(source, WindowFrameSource)
    assert source.window_title == "游戏窗口"


def test_backend_wgc_delegates_to_builder(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(source_factory, "_build_wgc", lambda cfg: sentinel)
    assert source_factory.build_frame_source(_window("wgc")) is sentinel


def test_backend_wgc_failure_propagates(monkeypatch):
    def _boom(cfg):
        raise RuntimeError("WGC 不可用")

    monkeypatch.setattr(source_factory, "_build_wgc", _boom)
    with pytest.raises(RuntimeError, match="WGC 不可用"):
        source_factory.build_frame_source(_window("wgc"))


def test_backend_auto_prefers_wgc(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(source_factory, "_build_wgc", lambda cfg: sentinel)
    assert source_factory.build_frame_source(_window("auto")) is sentinel


def test_backend_auto_falls_back_to_mss(monkeypatch, caplog):
    def _boom(cfg):
        raise RuntimeError("非 Windows")

    monkeypatch.setattr(source_factory, "_build_wgc", _boom)
    log = logging.getLogger("test-factory")
    with caplog.at_level(logging.INFO, logger="test-factory"):
        source = source_factory.build_frame_source(_window("auto"), logger=log)
    assert isinstance(source, WindowFrameSource)
    assert "降级 mss" in caplog.text
