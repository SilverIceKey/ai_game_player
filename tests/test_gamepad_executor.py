"""runtime/gamepad_executor.py / null_executor.py 单元测试（spec §10 输入设备适配）。

不触达真实 vgamepad / 键鼠：vgamepad 缺失路径用 sys.modules 屏蔽模拟，
可用路径注入假 backend。
"""
from __future__ import annotations

import sys
from unittest import mock

import pytest

from capture.action import NormalizedAction
from runtime.gamepad_executor import GamepadExecutor
from runtime.null_executor import NullExecutor


class TestGamepadExecutor:
    def test_missing_vgamepad_raises_clear_runtime_error(self):
        # sys.modules["vgamepad"] = None 使 import vgamepad 抛 ImportError
        with mock.patch.dict(sys.modules, {"vgamepad": None}):
            with pytest.raises(RuntimeError, match="ViGEmBus"):
                GamepadExecutor()

    def test_error_message_mentions_pip_install(self):
        with mock.patch.dict(sys.modules, {"vgamepad": None}):
            with pytest.raises(RuntimeError, match="pip install vgamepad"):
                GamepadExecutor()

    def test_injected_backend_placeholder_interface(self):
        """接口与 KeyboardMouseExecutor 对齐：execute 返回明细、release_all 不抛。"""
        ex = GamepadExecutor(keymap={"attack_light": "X"}, backend=object())
        assert ex.execute(NormalizedAction.neutral()) == "idle"
        detail = ex.execute(NormalizedAction(move_y=1.0, buttons=frozenset({"attack_light"})))
        assert "占位" in detail
        ex.release_all()  # 占位：不抛异常


class TestNullExecutor:
    def test_records_actions_without_side_effects(self):
        ex = NullExecutor()
        assert ex.execute(NormalizedAction.neutral()) == "idle"
        detail = ex.execute(NormalizedAction(move_y=1.0))
        assert "recorded" in detail
        assert len(ex.actions) == 2
        assert ex.actions[1].move_y == 1.0

    def test_release_all_counted(self):
        ex = NullExecutor()
        ex.release_all()
        ex.release_all()
        assert ex.release_all_calls == 2
