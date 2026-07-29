"""Ollama Provider：本地 Ollama 文本/视觉模型调用。

ollama SDK 延迟导入（首次调用时 import），Linux 开发机不安装也可跑全部测试。
模型与端点来自 configs/settings.yaml 的 llm 段；复盘用 vision_model（图文一起送）。
"""
from __future__ import annotations

from pathlib import Path


class OllamaProvider:
    """实现 llm.base.LLMProvider 契约，并扩展图文调用 complete_with_images。"""

    def __init__(self, model: str, base_url: str = "", vision_model: str = ""):
        self.model = model
        self.vision_model = vision_model or model
        self.base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import ollama  # 延迟导入：开发机不安装（pyproject optional-dependencies llm）
            except ImportError as exc:
                raise RuntimeError(
                    "未安装 ollama SDK；实机请执行: pip install ollama，"
                    "或 pip install 'ai-game-player[llm]'"
                ) from exc
            self._client = ollama.Client(host=self.base_url) if self.base_url else ollama.Client()
        return self._client

    def complete(self, prompt: str) -> str:
        """纯文本调用（llm.base.LLMProvider 契约），使用 llm.model。"""
        return self._chat(self.model, prompt, [])

    def complete_with_images(self, prompt: str, images: list[Path]) -> str:
        """图文调用（复盘帧分析），使用 llm.vision_model。"""
        return self._chat(self.vision_model, prompt, images)

    def _chat(self, model: str, prompt: str, images: list[Path]) -> str:
        if not model:
            raise RuntimeError("未配置模型：请在 configs/settings.yaml 的 llm 段设置 model/vision_model")
        client = self._get_client()
        message: dict = {"role": "user", "content": prompt}
        if images:
            message["images"] = [str(p) for p in images]
        try:
            resp = client.chat(model=model, messages=[message])
        except Exception as exc:
            raise RuntimeError(
                f"Ollama 调用失败（model={model}, base_url={self.base_url or '默认'}）: {exc}\n"
                f"请确认 Ollama 已启动（ollama serve）且模型已拉取: ollama pull {model}"
            ) from exc
        content = getattr(getattr(resp, "message", None), "content", None)
        if content is None and isinstance(resp, dict):  # 防御 SDK 返回结构变化
            content = (resp.get("message") or {}).get("content")
        return content or ""
