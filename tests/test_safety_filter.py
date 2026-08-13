"""runtime/safety_filter.py 单元测试（spec §39 + §40 + §26 + §47）。

override 键轮询与窗口焦点检查全部注入假实现，不触达真实 win32。
"""
from __future__ import annotations

import sys

import pytest

from capture.action import NormalizedAction
from config import SafetyConfig
from runtime.safety_filter import (
    MODE_AI_CONTROL,
    MODE_HUMAN_OVERRIDE,
    SafetyFilter,
    _default_focus_checker,
    _default_key_poller,
    vk_for_key,
)


class FakePoller:
    """可变按键状态 + 调用记录。"""

    def __init__(self):
        self.down = False

    def __call__(self, key: str) -> bool:
        return self.down


def _safety(**overrides) -> SafetyConfig:
    params = {
        "override_key": "F12",
        "stop_on_focus_lost": True,
        "max_button_hold_ms": 5000.0,  # 默认宽松，长按测试单独收紧
        "max_camera_delta": 0.5,
        "max_action_rate_hz": 1000.0,  # 默认不限频，避免干扰其他规则测试
    }
    params.update(overrides)
    return SafetyConfig(**params)


def _filter(safety: SafetyConfig | None = None, **kwargs) -> tuple[SafetyFilter, dict]:
    calls = {"release": 0, "clear": 0}
    sf = SafetyFilter(
        safety or _safety(),
        window_title="game",
        on_release=lambda: calls.__setitem__("release", calls["release"] + 1),
        on_clear=lambda: calls.__setitem__("clear", calls["clear"] + 1),
        **kwargs,
    )
    return sf, calls


def _nan_action() -> NormalizedAction:
    """构造轴向 NaN 的异常输出（绕过构造期 clamp，模拟模型异常直接落到 filter）。"""
    action = NormalizedAction.neutral()
    object.__setattr__(action, "camera_x", float("nan"))
    return action


class TestVkForKey:
    def test_function_keys(self):
        assert vk_for_key("F1") == 0x70
        assert vk_for_key("F12") == 0x7B

    def test_single_char(self):
        assert vk_for_key("a") == ord("A")

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            vk_for_key("LEFT_SHIFT")


class TestHumanOverrideToggle:
    """spec §26：override 键 toggle 切换 AI_CONTROL ⇄ HUMAN_OVERRIDE。"""

    def test_toggle_on_edge_blocks_actions_and_fires_dead_man_switch(self):
        poller = FakePoller()
        sf, calls = _filter(key_poller=poller, focus_checker=lambda: True)
        action = NormalizedAction(move_y=1.0, buttons=frozenset({"dodge"}))

        assert sf.check_environment().mode == MODE_AI_CONTROL
        assert sf.filter_action(action, 0) is action

        poller.down = True  # 按下沿：进入接管
        state = sf.check_environment()
        assert state.mode == MODE_HUMAN_OVERRIDE
        assert state.override_active
        assert calls["release"] >= 1 and calls["clear"] >= 1
        # 按住期间重复检查不得二次翻转
        assert sf.check_environment().mode == MODE_HUMAN_OVERRIDE
        # 接管期间全部动作被阻断
        assert sf.filter_action(action, 50_000) is None

        poller.down = False
        sf.check_environment()  # 必须观察到一次松开，边沿检测才能识别下一次按下
        poller.down = True  # 再按一次：恢复 AI 控制
        assert sf.check_environment().mode == MODE_AI_CONTROL
        assert sf.filter_action(action, 100_000) is action


class TestFocusLost:
    """spec §39：游戏窗口失焦立即 STOP ACTION。"""

    def test_focus_lost_blocks_and_releases(self):
        focused = {"ok": True}
        sf, calls = _filter(key_poller=lambda k: False, focus_checker=lambda: focused["ok"])
        action = NormalizedAction(move_y=1.0)
        assert sf.filter_action(action, 0) is action

        focused["ok"] = False
        state = sf.check_environment()
        assert state.focus_lost
        assert "失焦" in state.detail
        assert calls["release"] >= 1 and calls["clear"] >= 1
        assert sf.filter_action(action, 50_000) is None

    def test_stop_on_focus_lost_disabled(self):
        sf, _ = _filter(
            _safety(stop_on_focus_lost=False),
            key_poller=lambda k: False,
            focus_checker=lambda: False,
        )
        action = NormalizedAction(move_y=1.0)
        assert sf.filter_action(action, 0) is action


class TestButtonHoldLimit:
    """spec §39：单按钮连续按住超 max_button_hold_ms 自动释放。"""

    def test_auto_release_after_max_hold(self):
        sf, _ = _filter(
            _safety(max_button_hold_ms=100.0),
            key_poller=lambda k: False,
            focus_checker=lambda: True,
        )
        press = NormalizedAction(move_y=0.5, buttons=frozenset({"attack_light"}))

        assert sf.filter_action(press, 0).pressed("attack_light")  # 按下
        assert sf.filter_action(press, 50_000).pressed("attack_light")  # 按住 50ms
        # 按住 150ms > 100ms 上限：按钮被移除，其余轴保留
        out = sf.filter_action(press, 150_000)
        assert not out.pressed("attack_light")
        assert out.move_y == 0.5
        # 模型仍输出按住：持续强制释放，等模型先松手
        assert not sf.filter_action(press, 200_000).pressed("attack_light")

    def test_repress_allowed_after_model_releases(self):
        sf, _ = _filter(
            _safety(max_button_hold_ms=100.0),
            key_poller=lambda k: False,
            focus_checker=lambda: True,
        )
        press = NormalizedAction(buttons=frozenset({"attack_light"}))

        sf.filter_action(press, 0)
        assert not sf.filter_action(press, 150_000).pressed("attack_light")  # 超时释放
        sf.filter_action(NormalizedAction.neutral(), 200_000)  # 模型松手
        assert sf.filter_action(press, 250_000).pressed("attack_light")  # 允许重新按下


class TestCameraClamp:
    """spec §39：camera 轴单步超 max_camera_delta 截断。"""

    def test_camera_axes_clamped(self):
        sf, _ = _filter(key_poller=lambda k: False, focus_checker=lambda: True)
        action = NormalizedAction(camera_x=0.8, camera_y=-0.9, move_y=1.0)
        out = sf.filter_action(action, 0)
        assert out.camera_x == 0.5
        assert out.camera_y == -0.5
        assert out.move_y == 1.0  # 其他轴不受影响

    def test_within_limit_untouched(self):
        sf, _ = _filter(key_poller=lambda k: False, focus_checker=lambda: True)
        action = NormalizedAction(camera_x=0.3)
        assert sf.filter_action(action, 0) is action


class TestActionRateLimit:
    """spec §39：动作频率超 max_action_rate_hz 丢弃。"""

    def test_drops_actions_above_rate(self):
        # 40Hz → 最小间隔 25ms
        sf, _ = _filter(
            _safety(max_action_rate_hz=40.0),
            key_poller=lambda k: False,
            focus_checker=lambda: True,
        )
        action = NormalizedAction(move_y=1.0)
        assert sf.filter_action(action, 0) is action
        assert sf.filter_action(action, 10_000) is None  # 距上个放行 10ms < 25ms
        assert sf.filter_action(action, 20_000) is None  # 被丢弃的不重置计时
        assert sf.filter_action(action, 30_000) is action


class TestAbnormalOutput:
    """spec §39：异常模型输出（NaN / inf）自动丢弃。"""

    def test_nan_axis_dropped(self):
        sf, _ = _filter(key_poller=lambda k: False, focus_checker=lambda: True)
        assert sf.filter_action(_nan_action(), 0) is None

    def test_inf_axis_dropped(self):
        sf, _ = _filter(key_poller=lambda k: False, focus_checker=lambda: True)
        action = NormalizedAction.neutral()
        object.__setattr__(action, "move_x", float("inf"))
        assert sf.filter_action(action, 0) is None


class TestDeadManSwitch:
    """spec §40：Dead Man Switch = 释放全部输入 + 清空动作队列（注入回调执行）。"""

    def test_invokes_both_callbacks(self):
        sf, calls = _filter(key_poller=lambda k: False, focus_checker=lambda: True)
        sf.dead_man_switch()
        assert calls["release"] == 1
        assert calls["clear"] == 1

    def test_callback_failure_does_not_block_the_other(self):
        calls = {"clear": 0}

        def boom():
            raise RuntimeError("release failed")

        sf = SafetyFilter(
            _safety(),
            on_release=boom,
            on_clear=lambda: calls.__setitem__("clear", calls["clear"] + 1),
            key_poller=lambda k: False,
            focus_checker=lambda: True,
        )
        with pytest.raises(RuntimeError, match="dead man switch"):
            sf.dead_man_switch()
        assert calls["clear"] == 1  # 一个回调失败不阻断另一个


@pytest.mark.skipif(sys.platform == "win32", reason="仅验证非 Windows 安全默认")
class TestNonWindowsDefaults:
    def test_default_poller_never_pressed(self):
        assert _default_key_poller()("F12") is False

    def test_default_focus_checker_always_focused(self):
        assert _default_focus_checker("any title")() is True


def test_request_override_and_resume():
    """§26 数据闭环：编程模式请求在下一次 check_environment 生效（与 F12 toggle 并存）。"""
    sf, calls = _filter(key_poller=FakePoller(), focus_checker=lambda: True)
    assert sf.check_environment().mode == MODE_AI_CONTROL

    sf.request_override()
    state = sf.check_environment()
    assert state.mode == MODE_HUMAN_OVERRIDE and state.override_active
    assert calls["release"] >= 1 and calls["clear"] >= 1  # dead man switch 已触发

    sf.request_override()  # 幂等
    sf.request_resume()
    state = sf.check_environment()
    assert state.mode == MODE_AI_CONTROL and not state.override_active
