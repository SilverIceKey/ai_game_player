# PLAN-20260814-autopilot-voice-v1（待确认）

## 目标

AUTOPILOT 运行时增加语音播报能力：关键事件（启动/接管/恢复/超时/退出）+ 节流后的决策摘要，
TTS 走局域网 meloTts-server（`192.168.5.249:18103`），使用其纯标准库 SDK。

## 范围

- 只做 `app.autopilot` 链路；observe_train / train 不动。
- 不改 meloTts-server 项目任何文件（只读复制其 `sdk.py`）。
- 语音失败（服务不可达、无播放器）不得影响 AUTOPILOT 主链路：SDK 本身后台线程 +
  失败仅 stderr 打印，主流程零阻塞。

## 实施步骤

1. **复制 SDK**：`/usr/local/project/github/meloTts-server/sdk.py` → `runtime/tts_client.py`
   （原样复制，纯标准库，含 speak/stop/version；保留原 docstring 并注明来源）。
2. **配置**：`config.py` 新增 `VoiceConfig`（frozen dataclass）+ `_load_voice()`：
   - `voice.enabled`（bool，默认 false）
   - `voice.addr`（str，默认 `192.168.5.249:18103`）
   - `voice.speed`（float，默认 1.0）
   - `voice.language` / `voice.speaker`（str，默认空 = 服务端默认）
   - `voice.decision_interval_s`（float，默认 5.0；0 = 关闭决策播报）
   `Settings` 加 `voice` 字段；`configs/settings.example.yaml` 加对应段落与注释。
3. **播报器**：新增 `runtime/voice_announcer.py`：
   - `VoiceAnnouncer(client, decision_interval_s)`，薄封装：
     - `speak(text)`：直通 client.speak（新播报打断旧播报，SDK 语义）
     - `speak_decision(text)`：距上次决策播报 < interval 则丢弃
   - `format_action(action, keymap)`：把 Action 摘要成中文短语
     （move 方向 + 按下按键名，如 "前进+攻击"；无有效动作返回 None）。
4. **AUTOPILOT 接入**（`app/autopilot.py`，全部走注入 seam，测试不触网）：
   - `AutopilotSession.__init__` 新增可选参数 `announcer: VoiceAnnouncer | None = None`。
   - `start()`：播报 "自动驾驶已启动"。
   - `_track_mode()`：HUMAN_OVERRIDE_START → "已接管，交给你了"；
     AUTOPILOT_RESUME → "恢复 A I 控制"。
   - 推理超时分支（§47）：播报 "推理超时，AI 已暂停"。
   - `_dispatch_loop()`：每执行一个 AI 动作时调
     `announcer.speak_decision(format_action(...))`（仅 AI_CONTROL，节流由 announcer 保证）。
   - `stop()`：播报 "自动驾驶已退出"（block=True 短等，避免进程退出截断）+ `client.stop()`。
   - `main()`：`settings.voice.enabled` 时构建 `TTSClient(addr)` + `VoiceAnnouncer` 注入；
     启动时打印 voice 状态。
5. **测试**：
   - `tests/test_voice_announcer.py`：fake client 验证节流、format_action 文案、
     start/接管/恢复/超时/退出事件各触发一次播报（复用现有 AutopilotSession 假组件模式，
     参考 tests/test_app_autopilot.py）。
   - `tests/test_config_loader.py` 补 voice 段加载/默认值用例。
6. **文档收尾**：更新 `docs/progress/PROGRESS-20260814-autopilot-voice-v1.md`（新建）、
   `docs/agent-context.md`（主任务/改动/验证）。

## 验收方式

- `.venv/bin/python -m pytest -q` 全绿（含新用例）。
- `python -m compileall` 通过；settings.yaml 加载实测。
- 实机验证（用户侧）：游戏机上开 meloTts-server 可达时跑 autopilot，确认五类事件 + 决策播报出声。
  （开发机无法验证：非本机服务 + 无实机输入）

## 风险与待决项

- 决策播报内容是低层动作摘要（方向+按键），不是高层语义（"打 boss"），模型没有场景认知层；
  如需要语义级播报是另一个大需求，不在本轮。
- 播报依赖服务端在线与 Windows winsound / 本机播放器；失败仅 stderr，不影响闭环。
- `voice.addr` 默认值写用户给定的 `192.168.5.249:18103`，换网络环境改 settings.yaml 即可。
