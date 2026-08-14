# 交接上下文

## 当前状态

- 当前主任务：AUTOPILOT 实机输入来源隔离 + runtime Visual Token cache + epoch checkpoint 版本管理
- 当前阶段：代码与开发机验证完成，待 Windows/RTX2070S 实机验收；进度见 `docs/progress/PROGRESS-20260814-autopilot-runtime-checkpoints-v1.md`
- 当前结论：
  - 项目方向：端到端 Video-Action Policy（唯一权威规格 `docs/AI_Game_Player_SPEC_v1.0.md`）
  - 旧路线已全部删除归档（`docs/archive/20260729-legacy-runtime/`），不做向前兼容
  - 三链路：`app.observe_train`（采集 + --shadow）、`app.train`（训练 + 依赖消融 + gate 统计 + 段分布）、`app.autopilot`（持续采集闭环：自动接管/自动恢复/shadow inference/proposed 落盘/语音播报）
  - 架构：Token Transformer（ARCH_TAG=`token_transformer_v1`，唯一时序实现）；音频模态（§8.5）默认关
  - 数据闭环：同一条 episode 完整时间线（video + ai_proposed + executed + correction + marker）；段语义由 sample_builder 推导，失败段 AI 动作不回灌
  - 语音播报：`runtime/tts_client.py`（复制自 meloTts-server sdk.py，纯标准库）+ `runtime/voice_announcer.py`（事件直通 + 决策节流）；配置 `voice.*`（settings.yaml，默认关，addr 默认 192.168.5.249:18103）；播报点：启动/接管/恢复/超时/退出 + 节流决策摘要

## 本轮改动（2026-08-14 AUTOPILOT runtime/checkpoint）

- pynput Win32 hook 按 injected flag 过滤 AI 键鼠；AUTOPILOT 再校验 `source=human`；F12 与 `auto_takeover=false` 回退保留。
- TorchPolicy 加 Visual Token cache + `forward_tokens`，MemoryWriter 复用缓存；四项细分 latency；FP16 可配置、默认关闭。
- checkpoint 改为 `epochs/epoch-NNN` + `final` + 根汇总；root→final、显式 epoch、旧平铺目录兼容；registry 一次注册。
- 修复 PyTorch CPU eval fused fastpath 对当前 attention mask 产生 NaN；不改参数/state_dict。

## 上一轮改动（2026-08-14 语音播报）

- 计划 `docs/plans/PLAN-20260814-autopilot-voice-v1.md`；进度 `docs/progress/PROGRESS-20260814-autopilot-voice-v1.md`
- 新增 `runtime/tts_client.py`（SDK 复制）、`runtime/voice_announcer.py`（播报器 + format_action 文案）；`config.py` 加 `VoiceConfig`/`voice` 段；`app/autopilot.py` 接入 6 个播报点（可选注入）；`configs/settings.example.yaml` + `settings.yaml` 加 voice 段
- 测试 335 全绿（+10 新用例）

## 上一轮改动（2026-08-13 数据闭环）

- spec §26 重写（接管触发/自动恢复/持续采集/段分类规则）；计划 `docs/plans/PLAN-20260813-autopilot-data-loop-v1.md`；进度 `docs/progress/PROGRESS-20260813-autopilot-data-loop-v1.md`
- `safety_filter` 加编程模式切换；`autopilot` human_input_loop + shadow inference + marker；`episode_store` write_marker/write_proposed；`sample_builder` 段分类 + 失败剔除；config 加 auto_takeover/resume_idle_ms/pre_override_window_ms
- 测试 325 全绿（+5 新用例）

## 验证结果

- 已执行：`.venv/bin/python -m pytest -q` → 343 passed；定向覆盖输入隔离、cache/memory 复用、forward 等价、3 epochs/final/legacy loader。
- 未执行：Windows injected flag 实机、RTX2070S latency、FP16/FP32 对照、真实游戏闭环。
- 证据：`docs/reports/REPORT-20260814-autopilot-runtime-checkpoints-v1.md`

## 风险与限制

- Windows pynput/pydirectinput injected flag 实机行为仍待确认；确认前可保持 auto_takeover=false
- RTX2070S cache 后 latency 与 FP16 数值行为未测；FP16 默认关闭
- 旧 AUTOPILOT 数据无 marker，AI 动作会全部当 target（构建期有警告）
- 语音：TTS 服务离线时每条播报一次 stderr（不阻塞闭环）；决策播报为低层动作摘要，非语义级
- 训练/推理 memory 分布差、gate collapse 等上一轮风险不变

## 下一步

1. Windows 实机复验 injected event 不接管、真实输入/F12 可接管
2. RTX2070S 对比 cache 前后 p50/p95/p99 与四项 latency breakdown
3. FP16 只有实机 finite + FP32 行为对照通过后才允许默认开启

## 阅读顺序

1. `docs/progress/PROGRESS-20260814-autopilot-runtime-checkpoints-v1.md`
2. `docs/reports/REPORT-20260814-autopilot-runtime-checkpoints-v1.md`
3. `docs/AI_Game_Player_SPEC_v1.0.md`（§7/§16/§26/§29/§32/§33）
4. 代码入口：`capture/input/keyboard_mouse.py` → `app/autopilot.py`；`model/torch_policy.py` → `model/torch_model.py`；`train/trainer.py` → `model/checkpoint.py` / `train/registry.py`
