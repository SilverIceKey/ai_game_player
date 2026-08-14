# PROGRESS-20260814-autopilot-voice-v1

## 当前状态

- AUTOPILOT 语音播报已落地；实机发现 Python 3.13 + winsound `SND_MEMORY | SND_ASYNC` 不兼容问题，已修复。
- 计划文档：`docs/plans/PLAN-20260814-autopilot-voice-v1.md`；进度：本文件。

## 最近关键结论

- TTS 走 meloTts-server 局域网服务（`runtime/tts_client.py`，从其 sdk.py 原样复制，
  纯标准库；对方项目文件只读未动），地址配置化 `voice.addr`，当前默认/本机配置
  `192.168.5.249:18103`。
- 播报点（`app/autopilot.py`，经 `VoiceAnnouncer` 可选注入，None 即零开销零行为变化）：
  启动 "自动驾驶已启动" / 接管 "已接管，交给你了" / 恢复 "恢复 AI 控制" /
  推理超时（§47）"推理超时，AI 已暂停" / 退出 "自动驾驶已退出"（speak_exit 带 10s 上限等返回）。
- 决策播报：`_dispatch_loop` 每执行一个 AI 动作调 `speak_decision(format_action(...))`，
  节流间隔 `voice.decision_interval_s`（默认 5s，0=关闭）；文案为低层动作摘要
  （move 方向 + 按键中文名，如 "前进，轻击"），死区 0.2 仅用于文案。
- TTS 失败（服务不可达/无播放器）由 SDK 内部 stderr 打印兜底，AUTOPILOT 主链路零阻塞。

## 验证结果

- 已执行：`.venv/bin/python -m pytest -q` → 335 passed；`compileall` 通过。
- 实机联调：Windows Python 3.13 运行 `app.autopilot` 触发语音时，`winsound.PlaySound(memory, SND_ASYNC)` 报错 `RuntimeError: Cannot play asynchronously from memory`；已修复为临时文件 + `SND_ASYNC` 播放。
- 未执行：修复后实机再次出声验证（用户侧进行）。

## 阻塞项

- 无。

## 未证实风险

- Windows 游戏机端 winsound 播放效果与语速已修复（Python 3.13 下临时文件 + SND_ASYNC），
  待用户再次实机验证出声；服务端离线时每条播报一次 stderr 报错（不刷屏主日志，
  决策播报间隔外会持续重试，属预期降级）。
- 决策播报是低层动作摘要，非语义级（"在打 boss"）——模型无场景认知层，语义播报为独立大需求。

## 下一步动作

1. 游戏机实机跑 `python -m app.autopilot --game wukong`（settings.yaml voice.enabled=true），
   确认五类事件与决策摘要出声、音色/语速合适。
2. 实机验证后如需调文案/间隔，改 `voice.*` 配置或 `voice_announcer.py` 文案表即可。
