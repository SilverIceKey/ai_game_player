"""GamepadExecutor（spec §10：手柄输出接口）。

本轮仅为占位（计划文档第 1 节：手柄输出 ViGEm 本轮不做）：接口签名与
KeyboardMouseExecutor 对齐（execute/release_all），方便 app 层按
input_device 配置无差别装配；真实 ViGEm 调用逻辑留待后续实现。

手柄输出依赖 ViGEmBus 内核驱动（Windows）+ vgamepad 包，构造时即检测，
缺失则抛出明确 RuntimeError，而不是等到第一次 execute 才静默失败。
"""
from __future__ import annotations

from capture.action import NormalizedAction
from config import ExecutorConfig

_VIGEM_HINT = (
    "手柄输出需要 ViGEmBus 内核驱动（https://github.com/ViGEm/ViGEmBus，仅 Windows）"
    "并安装 Python 客户端：pip install vgamepad"
)


def _load_vgamepad():
    """延迟导入 vgamepad；隔离为模块级函数便于测试注入。"""
    import vgamepad

    return vgamepad


class GamepadExecutor:
    """手柄输出占位实现：构造时校验 ViGEm 可用性，方法体不触达真实设备。"""

    def __init__(
        self,
        keymap: dict[str, str] | None = None,
        params: ExecutorConfig | None = None,
        backend: object | None = None,
    ):
        if backend is None:
            try:
                _load_vgamepad()
            except ImportError as exc:
                raise RuntimeError(f"vgamepad 不可用：{_VIGEM_HINT}") from exc
        self._backend = backend  # 占位：后续接管真实 vgamepad 设备对象
        self._keymap = dict(keymap or {})
        self._params = params or ExecutorConfig()

    def execute(self, action: NormalizedAction) -> str:
        """占位：不做真实 ViGEm 调用，返回明细说明。"""
        if action.is_neutral():
            return "idle"
        return "gamepad: ViGEm 输出未实现（占位，仅校验依赖可用）"

    def release_all(self) -> None:
        """占位：无真实按住的输入可释放。"""
