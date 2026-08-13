# 交接上下文

## 当前状态

- 当前主任务：音频模态加入（spec §8.5）——代码完成，待游戏机实机验证；**开启 audio 后旧录制数据不可用，需重新采集**
- 当前阶段：用户已实机录制首段数据（无音频）；训练在 Windows 游戏机（CUDA）本地跑
- 当前结论：
  - 项目方向：端到端 Video-Action Policy（唯一权威规格 `docs/AI_Game_Player_SPEC_v1.0.md`）
  - 旧路线已全部删除归档（`docs/archive/20260729-legacy-runtime/`），不做向前兼容
  - 采集/推理/训练三链路全部可用：`app.observe_train`（采集 + --shadow）、`app.train`（训练）、`app.autopilot`（--checkpoint 加载真实模型）
  - 音频模态（§8.5）全链路落地：loopback 采集 → episode wav → log-mel → 可选 CNN 分支 → 推理 PCM 输入；默认关闭

## 本轮改动（2026-08-13 音频模态）

- spec §8.5 + §48 更新；计划 `docs/plans/PLAN-20260813-audio-modality-v1.md`
- `capture/audio.py`（WASAPI loopback，soundcard 延迟导入）、`dataset/episode_store.py`（每 episode wav + 时间窗切片读取）、`model/audio_features.py`（numpy log-mel）
- `VideoActionNet` 可选音频分支（checkpoint meta `audio` 段驱动重建）；`AudioRingBuffer` + InferenceWorker/两个 app 入口装配
- `config.py` 加 AudioConfig（默认关）；pyproject 加 soundcard（win32）
- 测试 310 全绿（新增 tests/test_audio.py 13 例）

## 验证结果

- 已执行：`pytest` 310 passed、compileall
- 未执行：WASAPI loopback 真实采集与带音频真实训练（开发机无声卡/无 GPU）
- 证据：`.venv/bin/python -m pytest -q` → 310 passed

## 风险与限制

- 独占音频模式游戏 loopback 可能抓不到（备选虚拟声卡）；音频↔帧同步偏差假设为常量未实测
- 训练后 audio 参数（sample_rate/mels/fft/hop）不可改（checkpoint 快照重建）
- 既有风险不变：mp4v 有损帧、DataLoader 吞吐、Action History mean-pool 表达力

## 下一步

1. 游戏机 `pip install soundcard` + settings.yaml 开 `audio.enabled: true` → OBSERVE_TRAIN 重录
2. `python -m app.train` 训练 → AUTOPILOT 实机验证（Phase 1 判据不变）
3. 不带音频的 Phase 1 也可先用旧数据跑通（audio 默认关，两条路互不阻塞）

## 阅读顺序

1. `docs/AI_Game_Player_SPEC_v1.0.md`（唯一权威规格，含 §8.5 Audio History）
2. `docs/plans/PLAN-20260813-audio-modality-v1.md`（音频设计决策）、`PLAN-20260813-training-pipeline-v1.md`（训练设计决策）
3. `docs/progress/PROGRESS-20260813-audio-modality-v1.md`（最新进度）
4. 代码入口：`app/train.py`、`app/observe_train.py`、`app/autopilot.py` → 契约 `capture/action.py`、`capture/audio.py`、`config.py`
