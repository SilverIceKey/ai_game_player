# PROGRESS-20260813-audio-modality-v1：音频模态加入（spec §8.5）

> 计划：`docs/plans/PLAN-20260813-audio-modality-v1.md`；规格：`docs/AI_Game_Player_SPEC_v1.0.md` §8.5

## 当前状态

- 当前主任务：音频模态加入（Video + Audio + Action → Future Action Chunk）——**代码完成，待游戏机实机验证**
- 当前结论：`audio.enabled=false` 默认关闭，全链路与现状完全一致；开启后 OBSERVE_TRAIN 录 wav、训练带音频分支、AUTOPILOT 音频进推理
- **已有 session 数据无音频：开启 audio 后必须重新采集（SessionDataset 检测到无音频会明确报错提示重录）**

## 本轮改动

- spec：§8.5 Audio History（采集方式/同步精度/特征参数/默认关闭）+ §48 capture/audio_capture
- 配置：`config.py` 加 `AudioConfig`（enabled/sample_rate=16000/mels=64/fft_size=400/hop_size=160）；settings.example.yaml 补 `audio:` 段；pyproject 加 `soundcard`（win32 only）
- 采集：`capture/audio.py`（AudioCapture 协议 + SoundcardLoopbackCapture WASAPI loopback + run_capture_loop 采集线程循环，块到达时刻 `now_us()` 打点、块起始=到达−时长）
- 存储：`dataset/episode_store.py` writer 加 `write_audio_chunk`（episode 级 wav，PCM s16 mono，stdlib wave），episode meta 记 audio_path/audio_start_us；reader 加 `audio_sample_rate()` / `load_audio_window()`（时间窗切片，窗外零填充）
- 特征：`model/audio_features.py` numpy log-mel（不引 librosa/torchaudio）+ `audio_window_us`（与 Video History 对齐）
- 模型：`VideoActionNet` 可选音频 CNN 分支（mel→2 层 CNN→128 维 concat 进 decoder）；checkpoint training_config 快照 `audio` 段驱动结构重建；`TorchPolicy.predict` 第三参 audio_pcm（内部转 mel，训练/推理共用同一函数）
- 数据/训练：`SessionDataset(audio=...)` 样本加 audio_mel 张量；`Trainer(audio=...)` forward 传音频
- 推理：`AudioRingBuffer`（runtime/ring_buffer.py，时间窗切片零填充）；`InferenceWorker` 注入 audio_buffer 时给 predict 传 PCM
- 装配：`app.observe_train` / `app.autopilot` 在 audio.enabled 时建采集线程（启动前探针验证 loopback 设备，失败明确退出）；带音频分支的 checkpoint 在未开启 audio 时启动即报错
- 测试：tests/test_audio.py 13 例（mel 数值、wav 往返、窗切片、ring buffer、假后端采集循环、模型分支、checkpoint 往返、数据集音频）

## 验证结果

- 已执行：`pytest` **310 passed**（297 + 音频 13）；compileall
- 未执行：WASAPI loopback 真实采集（开发机无声卡/非 Windows）；带音频的真实数据训练
- 证据：`.venv/bin/python -m pytest -q` → 310 passed in 10.28s

## 风险与限制

- soundcard loopback 在独占音频模式的游戏下可能抓不到声音——实机确认，备选虚拟声卡（如 VB-Cable）
- 音频↔帧同步偏差假设为常量（块到达时刻打点），秒级线索足够，未实测
- 音频播放端延迟（音箱/耳机）未标定
- settings 的 audio 参数（sample_rate/mels/fft/hop）训练后不可改——checkpoint 按快照重建，改了会结构不匹配或特征漂移

## 下一步

1. 游戏机：`pip install soundcard`，settings.yaml 设 `audio.enabled: true`
2. OBSERVE_TRAIN 重新采集数据（旧数据无音频不可用）
3. `python -m app.train` 训练带音频分支模型 → AUTOPILOT 实机验证
