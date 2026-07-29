"""截屏后端选择（mss/wgc/auto）与窗口提前台测试。

WGC 真实捕获路径仅 Windows 实机可达，这里覆盖：
配置新字段解析与非法值报错、backend 选择逻辑（monkeypatch 模拟成功/失败）、
foreground 在非 Windows 的行为。
"""
import sys
from pathlib import Path

import pytest
import yaml

from core.config import ConfigError
from core.perception.foreground import bring_to_foreground, list_window_handles
from core.perception.mss_source import WindowFrameSource
from core.perception.source_factory import build_frame_source
from games.wukong.adapter import WindowConfig, WukongConfig

WGC_CLS = "core.perception.wgc_source.WgcFrameSource"


def _window_cfg(**overrides) -> WindowConfig:
    base = {"title": "游戏窗口"}
    base.update(overrides)
    return WindowConfig(**base)


class _FakeWgc:
    """模拟 WGC 源：记录 start 握手是否被调用。"""

    def __init__(self, title, first_frame_timeout=5.0):
        self.title = title
        self.started = False

    def start(self):
        self.started = True

    def grab(self):
        raise NotImplementedError


class _BoomWgc:
    """模拟首帧握手失败的 WGC 源。"""

    def __init__(self, title, first_frame_timeout=5.0):
        pass

    def start(self):
        raise RuntimeError("WGC 首帧超时（5s）: 窗口未抓到画面")


# ---------- 配置解析与校验 ----------

def test_window_config_repo_defaults():
    cfg = WukongConfig.load("configs/wukong.yaml")
    assert cfg.window.capture_backend == "auto"
    assert cfg.window.foreground_on_start is True


def test_window_config_explicit_values():
    data = yaml.safe_load(Path("configs/wukong.yaml").read_text(encoding="utf-8"))
    data["window"]["capture_backend"] = "wgc"
    data["window"]["foreground_on_start"] = False
    cfg = WukongConfig.from_dict(data)
    assert cfg.window.capture_backend == "wgc"
    assert cfg.window.foreground_on_start is False


def test_window_config_invalid_backend_raises():
    data = yaml.safe_load(Path("configs/wukong.yaml").read_text(encoding="utf-8"))
    data["window"]["capture_backend"] = "dxgi"
    with pytest.raises(ConfigError, match="capture_backend"):
        WukongConfig.from_dict(data)


# ---------- backend 选择逻辑 ----------

def test_factory_forced_mss():
    source = build_frame_source(_window_cfg(capture_backend="mss"))
    assert isinstance(source, WindowFrameSource)


def test_factory_forced_wgc_unavailable_raises():
    # Linux 开发机：WgcFrameSource 初始化即抛明确错误，不做降级
    with pytest.raises(RuntimeError, match="Windows"):
        build_frame_source(_window_cfg(capture_backend="wgc"))


def test_factory_auto_falls_back_on_init_failure():
    # Linux 上 WGC 初始化自然失败 → auto 降级 mss
    source = build_frame_source(_window_cfg(capture_backend="auto"))
    assert isinstance(source, WindowFrameSource)


def test_factory_auto_prefers_wgc(monkeypatch):
    monkeypatch.setattr(WGC_CLS, _FakeWgc)
    source = build_frame_source(_window_cfg(capture_backend="auto"))
    assert isinstance(source, _FakeWgc)
    assert source.started  # auto 模式做了首帧握手


def test_factory_auto_falls_back_on_start_failure(monkeypatch):
    monkeypatch.setattr(WGC_CLS, _BoomWgc)
    source = build_frame_source(_window_cfg(capture_backend="auto"))
    assert isinstance(source, WindowFrameSource)


def test_factory_forced_wgc_start_failure_raises(monkeypatch):
    monkeypatch.setattr(WGC_CLS, _BoomWgc)
    with pytest.raises(RuntimeError, match="首帧超时"):
        build_frame_source(_window_cfg(capture_backend="wgc"))


# ---------- foreground 非 Windows 行为 ----------

def test_foreground_non_windows():
    if sys.platform == "win32":
        pytest.skip("本用例仅覆盖非 Windows 分支")
    assert list_window_handles() == []
    assert bring_to_foreground("任意窗口标题") is False
