"""AUTOPILOT 语音播报器（PLAN-20260814-autopilot-voice-v1）。

薄封装 TTSClient（runtime/tts_client.py，meloTts-server 局域网服务）：
- 事件播报直通（speak）：新播报打断旧播报（SDK 语义）；
- 决策播报节流（speak_decision）：按 decision_interval_s 丢弃过密播报，
  间隔外播报当前动作摘要（format_action）；
- TTS 失败由 SDK 内部 stderr 打印兜底，本层不抛异常，绝不影响 AUTOPILOT 主链路。
"""
from __future__ import annotations

import time

from capture.action import BUTTONS, NormalizedAction
from runtime.tts_client import TTSClient

# 按钮中文播报名（顺序对齐 capture.action.BUTTONS）
_BUTTON_ZH: dict[str, str] = {
    "attack_light": "轻击",
    "attack_heavy": "重击",
    "dodge": "闪避",
    "block": "格挡",
    "parry": "弹反",
    "jump": "跳跃",
    "interact": "互动",
    "heal": "回血",
    "skill_1": "技能一",
    "skill_2": "技能二",
    "skill_3": "技能三",
    "skill_4": "技能四",
    "lock_target": "锁定",
    "wait": "等待",
}

_MOVE_DEADZONE = 0.2  # 摇杆摘要死区（与 executor 死区解耦，仅用于文案）


def format_action(action: NormalizedAction) -> str | None:
    """把单步动作摘要成中文短语（如 "前进，轻击"）；无有效动作返回 None。"""
    parts: list[str] = []
    if action.move_y > _MOVE_DEADZONE:
        parts.append("前进")
    elif action.move_y < -_MOVE_DEADZONE:
        parts.append("后退")
    if action.move_x > _MOVE_DEADZONE:
        parts.append("右移")
    elif action.move_x < -_MOVE_DEADZONE:
        parts.append("左移")
    for name in BUTTONS:  # 固定顺序，文案稳定
        if name in action.buttons:
            parts.append(_BUTTON_ZH[name])
    return "，".join(parts) if parts else None


class VoiceAnnouncer:
    """AUTOPILOT 语音播报：事件直通 + 决策节流。"""

    def __init__(
        self,
        client: TTSClient,
        *,
        decision_interval_s: float = 5.0,
        speed: float = 1.0,
        language: str = "",
        speaker: str = "",
    ):
        self._client = client
        self._interval = decision_interval_s
        self._speed = speed
        self._language = language or None
        self._speaker = speaker or None
        self._last_decision = 0.0

    def speak(self, text: str) -> None:
        """事件播报：直通，打断上一条。"""
        self._client.speak(text, speed=self._speed, language=self._language,
                           speaker=self._speaker)

    def speak_decision(self, text: str | None) -> None:
        """决策播报：空文本或 interval 内直接丢弃；interval<=0 表示关闭。"""
        if not text or self._interval <= 0:
            return
        now = time.monotonic()
        if now - self._last_decision < self._interval:
            return
        self._last_decision = now
        self.speak(text)

    def speak_exit(self, text: str, timeout_s: float = 10.0) -> None:
        """退出播报：等待请求线程返回（带超时），让最后一句尽量出声。"""
        thread = self._client.speak(text, speed=self._speed, language=self._language,
                                    speaker=self._speaker)
        try:
            thread.join(timeout_s)
        except KeyboardInterrupt:
            # 用户 Ctrl+C 退出时不必等 TTS 返回，继续收尾（writer.close 等）
            pass
