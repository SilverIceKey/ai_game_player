# 音频模态加入计划（Video + Audio + Action → Future Action Chunk）

## 背景

用户要求：视频之外，游戏声音（Boss 抬手音效、受击音、环境音）也作为模型输入。spec 已更新 §8.5 Audio History（默认关闭的可选模态），本计划按该节落地。

**重要前提：已有录制数据不含音频，开启后需重新采集（OBSERVE_TRAIN 重录）。**

## 设计决策

1. **采集源**：Windows WASAPI loopback（录游戏音频输出，不需要麦克风权限/驱动），库用 `soundcard`（纯 Python、loopback API 简洁，延迟导入）。参数：16000Hz / mono / float32。
2. **同步（§11 一致性）**：音频块到达时刻用 `now_us()` 打点，块起始时间 = 到达时刻 − 块时长。音频对动作的指导是秒级线索（Boss 前摇音效），不像帧↔输入要求毫秒级，此近似足够；偏差假设记录为常量。
3. **存储（§20 扩展）**：每 episode 一个 `audio/episode_000.wav`（stdlib `wave` 写 PCM s16，16k mono，不引 soundfile）；episode meta 记 `audio_start_us`，样本时间 → 字节偏移直接换算；manifest.capture 加 `audio: {sample_rate, channels}`。`audio_enabled=false` 时目录与字段完全不出现（与现格式一致）。
4. **特征**：log-mel 频谱，**numpy 实现**（STFT + mel 滤波器组 + log，不引 librosa/torchaudio）：64 mels，窗长 25ms、hop 10ms。样本音频窗 = 与 Video History 对齐的过去窗口（≈1330ms，随 history_frames/sample_fps 换算）。
5. **模型**：VideoActionNet 增加可选音频分支：log-mel (64×T) → 两层小 CNN → 128 维向量 → 与 GRU 最后隐状态 + action 向量 concat 进 Policy Decoder。`audio_enabled` 写入 checkpoint 的 training_config，加载时按 meta 重建结构——有/无音频的 checkpoint 结构不同但不混淆（由 meta 驱动）。
6. **推理侧**：AUTOPILOT 增加 audio 采集线程 + `AudioRingBuffer`（按时间窗切片）；`audio_enabled` 的模型在无音频设备环境下启动即明确报错。OBSERVE_TRAIN 同样由配置开关。
7. **配置**：settings 新增 `audio: {enabled: false, sample_rate: 16000, mels: 64, fft_size: 400, hop_size: 160}`（默认关，显式开启；游戏 yaml 不动）。

## 改动清单

| 层 | 文件 | 改动 |
|---|---|---|
| spec | `docs/AI_Game_Player_SPEC_v1.0.md` | §8.5 Audio History + §48 capture/audio_capture（已完成） |
| 配置 | `config.py` + `configs/settings.example.yaml` | `AudioConfig`（enabled/sample_rate/mels/fft_size/hop_size） |
| 采集 | `capture/audio.py`（新） | `AudioCapture` 协议 + `SoundcardLoopbackCapture`（soundcard 延迟导入）+ 测试用注入假后端 |
| 存储 | `dataset/episode_store.py` | writer 加 `begin_audio(start_us)` / `write_audio_chunk(pcm_f32)` / episode 收尾落 wav；reader 加音频读取（按时间窗切片返回 float32） |
| 特征 | `model/audio_features.py`（新） | numpy log-mel（纯函数可测）+ 音频窗切片对齐 |
| 模型 | `model/torch_model.py` | 可选音频 CNN 分支（audio_enabled + mel 参数进构造） |
| 数据 | `train/dataset.py` | 样本加 audio mel 张量（audio_enabled 时） |
| 训练 | `train/trainer.py` | training_config 快照加 audio 段；forward 传音频 |
| 推理 | `runtime/ring_buffer.py` 加 `AudioRingBuffer`；`model/torch_policy.py` predict 加音频窗；`model/policy.py` 协议文档更新 | |
| 应用 | `app/observe_train.py` / `app/autopilot.py` | audio.enabled 时装配音频采集/缓冲 |
| 依赖 | `pyproject.toml` | + `soundcard` |
| 测试 | tests/ | mel 数值、wav 往返、时间窗切片、模型前向（带/无音频）、checkpoint 往返（含 audio 配置）、CLI 开关装配 |

## 明确不做（本轮）

- 音频播放端标定（音箱/耳机延迟测量）——记录偏差假设为常量
- 多声道/空间音频、语音内容识别（§1 不做语义理解）
- 音频增强（降噪/增益归一）——先原始信号

## 实施步骤

1. spec §8.5 + §48 更新（已完成）
2. config + pyproject + settings.example.yaml
3. capture/audio.py + dataset/episode_store.py 音频读写
4. model/audio_features.py（log-mel）
5. torch_model/torch_policy/dataset/trainer 接音频分支
6. app 两个入口装配
7. 测试 → pytest 全绿 + compileall
8. 文档收口：progress + agent-context + README（重新采集提示）

## 验证与风险

- 开发机验证：纯函数与合成音频单测（无声卡）；**WASAPI loopback 真实采集只能游戏机验证**
- 风险：soundcard loopback 在独占音频模式游戏下可能抓不到（实机确认；备选方案是虚拟声卡）；音频与帧的真实同步偏差未实测；用户已有 session 数据无音频需重录
