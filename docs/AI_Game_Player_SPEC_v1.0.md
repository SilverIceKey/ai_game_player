# AI Game Player SPEC v1.0

> 面向类魂 / 第三人称动作游戏的端到端视觉控制 Agent

---

## 1. 项目定位

### 1.1 项目目标

`AI Game Player` 的目标不是构建一个“理解游戏世界全部语义”的通用游戏 Agent，也不是通过 OCR、目标检测、世界状态机、行为树逐层拆解游戏逻辑。

第一阶段目标是验证一条更直接的路线：

```text
Video History
+
Action History
+
Optional Memory / Goal
        ↓
   Video-Action Policy
        ↓
 Future Action Chunk
```

即：

> 通过观察玩家真实游玩过程中的连续画面和同步操作记录，训练一个专用时序控制模型，使其能够直接根据当前视觉上下文输出下一段游戏操作，并在真实游戏环境中闭环运行。

第一阶段重点针对：

- 《黑神话：悟空》
- 类魂游戏
- 第三人称动作游戏
- 半线性 / 线性 3A 动作游戏

不以复杂开放世界、MMO、SLG、卡牌、强文字 UI 游戏作为第一阶段目标。

---

## 2. 核心假设

项目需要验证的核心假设是：

\[
P(A_{t:t+n} \mid V_{t-k:t}, A_{t-k:t-1}, M_t, G_t)
\]

是否已经包含足够的信息，使模型可以学习人类玩家在类魂游戏中的主要操作策略。

其中：

- `V`：最近一段时间的游戏视频。
- `A_history`：最近一段时间玩家执行过的动作。
- `M`：可选的长期记忆 / latent memory。
- `G`：可选的高层目标。
- `A_future`：未来短时间的动作序列。

第一阶段允许模型“不知道”：

- 敌人叫什么。
- Boss 技能叫什么。
- 血条是什么意思。
- 哪个动作叫闪避。
- 地图叫什么。
- 世界观是什么。

模型只需要学习：

> 在这种连续视觉状态和最近操作历史下，人类下一步通常如何操作。

---

## 3. 适用场景

### 3.1 优先支持

第一阶段优先覆盖：

- 局部寻路。
- 跑图。
- 探索。
- 道路跟随。
- 简单岔路选择。
- 采集。
- 宝箱 / 可交互物交互。
- 普通敌人战斗。
- 精英怪战斗。
- Boss 战。
- 闪避。
- 格挡 / 弹反。
- 喝药。
- 技能释放。
- 死亡。
- 复活。
- 重试。
- 过场等待。
- 加载等待。

### 3.2 暂不优先

以下问题不作为 MVP 必须解决项：

- 剧情理解。
- NPC 长对话理解。
- 复杂任务系统。
- 支线任务规划。
- 装备 Build。
- 技能树规划。
- 大地图全局规划。
- 高自由度开放世界长期目标。
- 游戏知识库。
- Wiki / 攻略检索。
- OCR 驱动的强符号策略。
- 游戏内存读取。
- 游戏进程注入。
- 协议逆向。
- 绕过反作弊机制。

---

## 4. 游戏类型边界

### 4.1 适合端到端视觉控制

适合程度高：

```text
类魂
第三人称动作
Boss Rush
格斗
赛车
平台跳跃
部分 FPS
线性动作游戏
```

共同特点：

> 当前几秒的视觉状态，对下一步动作具有很强决定性。

---

### 4.2 需要额外高层策略

例如：

- 《艾尔登法环》
- GTA
- 《荒野大镖客 2》
- 大型开放世界 RPG

这些游戏仍然可以使用 Video-Action Policy 做低层控制，但通常还需要：

```text
Long-term Memory
+
Goal Conditioning
+
Slow Planner
```

---

### 4.3 不适合作为纯视觉端到端首个目标

例如：

- 《梦幻西游》
- SLG
- 卡牌
- MMO
- 经营模拟
- 强文字 / 强 UI 游戏

这类游戏更适合：

```text
ROI
+
OCR
+
Structured State
+
Policy
```

而不是强行从 RGB 学习所有符号信息。

---

# 5. 顶层运行模式

系统仅定义两种顶层模式：

```text
OBSERVE_TRAIN
AUTOPILOT
```

---

## 5.1 OBSERVE_TRAIN

中文名称：

> 观察训练模式

定义：

> 用户自己玩，系统只观察、记录、构造样本、训练候选模型，不主动控制游戏。

完整流程：

```text
Human Player
     │
     ├────────────→ Keyboard / Mouse / Gamepad
     │
     ▼
    Game
     │
     ▼
Screen Capture
     │
     ├────────→ Video Stream
     │
Input Capture
     │
     └────────→ Action Stream
                    │
                    ▼
              Timestamp Sync
                    │
                    ▼
                Episode Store
                    │
                    ▼
               Sample Builder
                    │
                    ▼
                Replay Buffer
                    │
                    ▼
                   Trainer
                    │
                    ▼
              Candidate Model
                    │
                    ▼
                Evaluation
                    │
                    ▼
               Model Registry
```

---

## 5.2 AUTOPILOT

中文名称：

> 自动游玩模式

定义：

> 系统加载一个冻结的 Active Model，在真实游戏画面上进行闭环推理和输入控制。

流程：

```text
Game
 │
 ▼
Screen Capture
 │
 ▼
Frame Ring Buffer
 │
 ▼
Preprocess
 │
 ▼
Video-Action Policy
 │
 ▼
Action Chunk
 │
 ▼
Safety Filter
 │
 ▼
Input Executor
 │
 ▼
Game
```

形成：

```text
Observation
→
Model
→
Action
→
Environment Change
→
New Observation
```

---

# 6. 不允许的训练方式

第一版不允许：

```text
每采一帧
→
立即 backward()
```

原因：

- 与游戏抢 GPU。
- 影响游戏 FPS。
- 破坏采集时序。
- 产生 latency jitter。
- 训练过程不可复现。
- Active Model 行为可能不稳定。

推荐策略：

```text
战斗 / 游戏进行中
→
只采集

Episode 结束 / 暂停 / 加载阶段
→
构造样本 / 训练
```

如果后续存在独立训练 GPU，则允许异步训练，但 Active Model 仍必须冻结。

---

# 7. 模型生命周期

模型必须分成：

```text
Active Model
Candidate Model
```

禁止训练后直接覆盖当前模型。

流程：

```text
Dataset Version N
       ↓
Train
       ↓
Candidate Model
       ↓
Offline Evaluation
       ↓
Closed-loop Evaluation
       ↓
Pass?
  │        │
 Yes       No
  │        │
  ▼        ▼
Promote   Reject
  │
  ▼
Active Model N+1
```

模型切换只能发生在：

```text
Episode Boundary
```

不能在一次战斗 / 控制过程中热切模型。

---

# 8. MVP 输入

## 8.1 Video History

模型不能使用：

```text
Single Frame
→
Action
```

必须使用时间窗口：

```text
Frame[t-k]
Frame[t-k+1]
...
Frame[t]
```

原因：

单帧无法稳定表达：

- Boss 正在抬手还是收招。
- 玩家正在移动还是停止。
- 敌人正在靠近还是远离。
- 当前是否刚刚闪避。
- 当前 combo 处于哪个阶段。
- 当前是否刚刚受击。

---

## 8.2 Action History

模型输入还必须包含：

```text
Action[t-m:t-1]
```

作用：

- 表达当前持续按键。
- 表达攻击连段上下文。
- 表达闪避状态。
- 表达镜头运动状态。
- 减少视觉歧义。

---

## 8.3 Optional Memory

完整游戏流程中，仅靠最近几秒 Video 不足以解决：

- 来过这个岔路没有。
- 刚才从哪里过来。
- 是否已经采集。
- 当前处于探索还是回退。
- 某条路是否刚刚失败。

因此后续模型需支持：

```text
Latent Memory / Recurrent State
```

例如：

\[
h_t = f(V_t, A_{t-1}, h_{t-1})
\]

第一阶段允许 Memory 不启用。

Boss 战 MVP 可以只做短时序。

---

## 8.4 Optional Goal

对于长程场景，可以增加非常轻量的目标 conditioning：

```text
EXPLORE
COMBAT
FOLLOW_PATH
INTERACT
RECOVER
```

不要求一开始实现复杂 Planner。

---

## 8.5 Optional Audio History

游戏音频（Boss 抬手音效、受击音、环境音）包含画面之外的秒级线索，作为可选模态加入输入：

```text
Audio History = 与 Video History 对齐的过去窗口音频
```

约束：

- 采集方式为系统音频输出回录（WASAPI loopback），不依赖麦克风。
- 单声道、16kHz 采样率足够表达游戏线索音，不追求高保真。
- 音频与视频使用同一时钟（`now_us()`）打点；音频块到达时刻打点，块起始时间 = 到达时刻 − 块时长。音频是秒级线索，不要求帧↔输入那样的毫秒级同步精度。
- 模型输入形式为 log-mel 频谱（64 mels，窗长 25ms、hop 10ms）。
- 默认关闭，由配置显式开启；开启后历史录制数据（无音频）不可用于训练，需重新采集。

第一阶段允许 Audio 不启用。

---

# 9. 动作空间

内部动作表示必须与实际键位解耦。

例如：

```json
{
  "move_x": 0.0,
  "move_y": 0.0,
  "camera_x": 0.0,
  "camera_y": 0.0,

  "attack_light": false,
  "attack_heavy": false,

  "dodge": false,
  "block": false,
  "parry": false,

  "jump": false,
  "interact": false,

  "heal": false,

  "skill_1": false,
  "skill_2": false,
  "skill_3": false,
  "skill_4": false,

  "lock_target": false,
  "wait": false
}
```

实际键位由：

```text
InputAdapter
```

负责映射。

例如：

```text
heal
→
Keyboard F
```

模型不直接学习 `F` 的语义。

---

# 10. 输入设备适配

支持：

```text
KeyboardMouseAdapter
GamepadAdapter
```

统一转换为：

```text
NormalizedAction
```

手柄数据优先使用连续轴：

```text
LX
LY
RX
RY
LT
RT
Buttons
```

键鼠数据：

- WASD 转换成二维移动向量。
- 鼠标 delta 归一化。
- 鼠标按钮转换成离散 Action。

---

# 11. 时间同步

这是整个项目最高优先级之一。

所有时间必须使用同一 monotonic clock。

Windows 推荐：

```text
QueryPerformanceCounter
```

统一覆盖：

```text
Frame Capture
Input Capture
Inference
Action Dispatch
Training Sample
```

必须能够精确回答：

> 某一帧对应玩家哪一个操作？

目标：

```text
同步误差 < 10ms
```

理想：

```text
< 5ms
```

---

# 12. Human Reaction Delay

行为克隆存在一个重要问题：

```text
Boss 出招
↓
人看到
↓
人反应
↓
按闪避
```

人类自身存在约几十到数百毫秒反应延迟。

因此训练样本必须支持：

```yaml
action_label_offset_ms: configurable
```

实验搜索至少包括：

```text
0
50
100
150
200
250 ms
```

最终通过 closed-loop 成绩决定，不允许拍脑袋固定。

---

# 13. 采集参数建议

第一版建议：

```yaml
capture:
  source_fps: 60

model:
  sample_fps: 12
  history_frames: 16
  history_duration_ms: ~1330

prediction:
  action_step_ms: 50
  future_action_steps: 4
  horizon_ms: 200
```

这些不是固定最优值，只是第一版搜索起点。

---

# 14. 模型分辨率

建议起点：

```yaml
model_input:
  width: 384
  height: 216
```

后续比较：

```text
384×216
448×252
512×288
```

不建议直接用原生 1080P 输入 Policy。

也不建议第一版压到极低分辨率后再假设模型能识别所有 Boss 前摇。

---

# 15. Action Chunking

模型不建议每次只预测一个瞬时操作。

推荐：

```text
一次 inference
→
未来 4 个 action step
```

例如：

```text
t+0ms
t+50ms
t+100ms
t+150ms
```

作用：

- 降低推理调用频率。
- 降低动作抖动。
- 学习短时间连续操作。
- 减弱单帧预测误差。
- 更适合攻击 combo / 闪避节奏。

---

# 16. 模型架构

第一阶段推荐：

```text
Video Frames
    │
    ▼
Visual Encoder
    │
    ▼
Visual Tokens
    │
    ▼
Temporal Encoder
    ▲
    │
Action History
    │
    ▼
Policy Decoder
    │
    ▼
Action Chunk
```

不要求显式：

- Object Detection。
- OCR。
- Boss Classification。
- Pose Estimation。
- Depth Estimation。
- Skill Recognition。
- HP 数值解析。

---

# 17. 模型参数规模

第一阶段不追求大模型。

建议初始范围：

```text
Visual Encoder:
50M ~ 150M

Temporal / Policy:
100M ~ 300M

Total:
150M ~ 400M
```

这只是初始工程搜索区间。

如果模型效果差，排查顺序必须是：

```text
Timestamp
↓
Labels
↓
Dataset
↓
Action Representation
↓
Class Imbalance
↓
Temporal Window
↓
Resolution
↓
Loss
↓
Distribution Shift
↓
Model Capacity
```

禁止第一反应：

```text
模型不行
→
直接上 7B
```

---

# 18. Visual Encoder 训练策略

第一版建议：

```text
Pretrained Visual Encoder
+
Temporal Policy
```

训练阶段：

```text
Stage 1
Freeze Visual Backbone

Stage 2
Unfreeze Last Blocks

Stage 3
Optional Full Fine-tuning
```

不要从随机初始化训练完整视觉系统。

---

# 19. Action Head

建议拆 Head：

```text
Movement Head
Camera Head
Combat Head
Utility Head
```

---

## 19.1 Movement Head

输出：

```text
move_x
move_y
```

范围：

```text
[-1, 1]
```

可测试：

- Regression。
- Discretized bins。

---

## 19.2 Camera Head

鼠标 / 右摇杆属于连续空间。

不建议只用 MSE Regression。

因为可能出现：

```text
左转 = -0.8
右转 = +0.8

模型不确定
→
0
```

建议优先实验：

```text
Discretized Distribution
```

---

## 19.3 Button Head

例如：

```text
attack
dodge
heal
jump
interact
skill
```

使用：

```text
multi-label binary heads
```

不能将所有组合动作编码成一个巨大 action class。

---

# 20. 数据结构

推荐 Session：

```text
sessions/
└── 20260812_001/
    ├── manifest.json
    ├── video/
    ├── frames.idx
    ├── actions.bin
    ├── events.jsonl
    └── telemetry.jsonl
```

---

## 20.1 Session Manifest

```json
{
  "session_id": "20260812_001",
  "mode": "OBSERVE_TRAIN",
  "game": "black_myth_wukong",

  "capture": {
    "width": 1920,
    "height": 1080,
    "fps": 60
  },

  "input_device": "gamepad",
  "dataset_version": "dataset-v001",

  "labels": {
    "quality": "unreviewed"
  }
}
```

---

## 20.2 Frame Record

```json
{
  "frame_id": 10086,
  "timestamp_us": 87230199210,
  "video_offset": 193921
}
```

---

## 20.3 Action Record

```json
{
  "timestamp_us": 87230201120,

  "move_x": 0.0,
  "move_y": 1.0,

  "camera_x": 0.13,
  "camera_y": -0.02,

  "attack_light": false,
  "dodge": true,
  "heal": false,
  "interact": false
}
```

高频 Action Record 实际保存建议使用 binary format。

JSON 仅用于调试和导出。

---

# 21. Episode

整个录像必须切 Episode。

Episode 示例：

```text
土地庙出发
→
探索
→
小怪
→
采集
→
继续移动
→
Boss
→
死亡
```

也可以切成多个短 Episode。

MVP 如果自动检测不稳定，可以由人工：

```text
START EPISODE
STOP EPISODE
```

不要为了全自动提前增加不稳定感知模块。

---

# 22. Training Sample

训练样本逻辑：

```text
Observation:
  frames[t-k:t]

Action History:
  actions[t-m:t-1]

Target:
  actions[t:t+n]
```

例如：

```text
16 frames
+
最近动作
+
未来 4 个 action step
```

---

# 23. Behavior Cloning

第一阶段训练采用：

```text
Supervised Imitation Learning
```

Loss：

\[
L =
L_{move}
+
\lambda_c L_{camera}
+
\lambda_b L_{button}
+
\lambda_t L_{temporal}
\]

所有 Loss 权重必须配置化并记录实验。

---

# 24. Class Imbalance

数据中大量时间可能是：

```text
移动
等待
普通攻击
```

而：

```text
闪避
喝药
弹反
特殊技能
```

出现频率低但价值高。

必须支持：

```text
Class Weighting
Balanced Sampling
Rare Action Oversampling
```

不能使用整体 accuracy 作为主指标。

---

# 25. Recovery Dataset

只训练“高手正常操作”不够。

AI 一旦犯错，会进入 Human Dataset 中很少出现的状态。

必须专门保留：

```text
卡墙
镜头丢失
闪避失败
攻击落空
被连击
低血
喝药失败
走错路
掉下平台
找不到出口
死亡
```

这些属于：

```text
Recovery Data
```

---

# 26. Human Override

AUTOPILOT 必须支持即时人工接管。

例如：

```text
F12
```

或者手柄组合键。

触发：

```text
AI_CONTROL
↓
HUMAN_OVERRIDE
```

必须立即：

- 清空 Action Queue。
- 释放所有按键。
- 停止 AI 输入。
- 玩家接管。

玩家接管后的操作继续记录为：

```text
source = correction
```

---

# 27. DAgger 式闭环

长期训练：

```text
Human Demonstration
        ↓
Model V1
        ↓
AI Play
        ↓
Failure
        ↓
Human Takeover
        ↓
Correction Data
        ↓
Model V2
```

持续重复。

目标不是：

> 模型学会“所有正确状态”。

而是逐渐覆盖：

> 模型自己会进入的状态分布。

---

# 28. Replay Buffer

Replay Buffer 必须同时包含：

```text
Historical
Recent
Correction
Rare Event
```

示例初始采样权重：

```yaml
sampling:
  historical: 0.50
  recent: 0.25
  correction: 0.20
  rare: 0.05
```

此比例只是起点，需要实验确定。

---

# 29. Dataset Versioning

必须：

```text
dataset-v001
dataset-v002
dataset-v003
```

每个模型记录：

```text
Model Version
Dataset Version
Code Commit
Training Config
Evaluation Result
```

否则不可复现。

---

# 30. AUTOPILOT Runtime

推荐线程结构：

```text
Capture Thread
      │
      ▼
Frame Queue
      │
      ▼
Preprocess Worker
      │
      ▼
History Ring Buffer
      │
      ▼
Inference Worker
      │
      ▼
Action Scheduler
      │
      ▼
Safety Filter
      │
      ▼
Input Executor
```

禁止把：

```text
capture
inference
input
```

全部阻塞在同一个线程。

---

# 31. 延迟目标

第一版工程目标：

```text
Capture:
< 8ms

Preprocess:
< 5ms

Inference:
< 30ms

Decode:
< 2ms

Input Dispatch:
< 5ms
```

端到端目标：

```text
P50 < 50ms
P95 < 80ms
P99 < 120ms
```

这些是目标，不代表当前硬件必然达到。

---

# 32. 延迟可观测性

必须记录：

```text
P50
P90
P95
P99
MAX
```

不能只看平均延迟。

同时记录：

```text
Capture FPS
Dropped Frames
Frame Age
Queue Delay
Inference FPS
GPU Utilization
VRAM
Temperature
```

---

# 33. Runtime 日志

每次 inference 至少记录：

```json
{
  "timestamp_us": 87230201120,
  "model_version": "model-v017",
  "observation_id": "obs-...",
  "frame_age_ms": 16,
  "queue_delay_ms": 4,
  "inference_ms": 27.4,
  "action": {},
  "action_confidence": {},
  "mode": "AUTOPILOT"
}
```

---

# 34. Dataset 可观测性

至少统计：

```text
Total Hours
Total Episodes

Human Episodes
AI Episodes
Correction Episodes

Death Episodes
Success Episodes

Move Ratio
Idle Ratio
Attack Count
Dodge Count
Heal Count
Interact Count

Action Distribution
Rare Action Count
```

---

# 35. Offline Evaluation

固定冻结 Eval Set：

```text
eval/
├── traversal
├── narrow_path
├── interaction
├── collection
├── normal_combat
├── low_hp
├── boss_combo
├── camera_lost
├── cornered
├── healing
├── recovery
└── death_restart
```

训练集不能覆盖固定评估集。

---

# 36. Offline Metrics

至少：

```text
Movement Error
Camera Error

Dodge Precision / Recall
Heal Precision / Recall
Interact Precision / Recall
Attack Precision / Recall

Action Sequence Accuracy
Temporal Consistency
```

禁止只看 Overall Accuracy。

---

# 37. Closed-loop Evaluation

真正决定模型是否可用的指标：

```text
Average Autonomous Time
Manual Takeover Rate

Distance Travelled
Stuck Time

Successful Interaction Count
Successful Collection Count

Enemy Kill Count
Boss Phase Reached
Boss Victory Rate

Damage Taken / Minute
Death Rate
Recovery Success Rate
```

第一阶段主指标：

```text
Autonomous Duration
Manual Takeover Rate
Boss / Combat Success
Stuck Rate
```

---

# 38. Shortcut Learning

模型可能学习错误视觉 shortcut。

例如：

真正目标：

```text
低血
→
Heal
```

错误学习：

```text
屏幕发红
→
Heal
```

必须构造反例：

```text
HP高 + 红色特效
HP低 + 无红屏
HP低 + Boss正在攻击
HP低 + 安全状态
```

对关键能力建立 counter-example eval。

---

# 39. Safety Filter

模型不能直接无限制控制 OS。

必须限制：

- 游戏窗口必须 Foreground。
- 禁止系统快捷键。
- 禁止 Win Key。
- 禁止 Alt+F4。
- 禁止应用切换。
- 最大连续按键时间。
- 最大鼠标移动量。
- 最大动作频率。
- 异常模型输出自动丢弃。

如果：

```text
Game Window Lost Focus
```

立即：

```text
STOP ACTION
```

---

# 40. Dead Man Switch

必须存在紧急停止。

触发后：

```text
Clear Action Queue
Release All Keys
Release Mouse Buttons
Stop Input Executor
```

避免按键卡死。

---

# 41. Shadow Mode

OBSERVE_TRAIN 必须包含一个子状态：

```text
SHADOW
```

用户仍然控制。

AI 实时预测：

```text
AI 现在想做什么
```

但：

```text
不执行
```

用于验证：

- 推理延迟。
- Action 对齐。
- 模型意图。
- 关键动作召回。

进入 AUTOPILOT 前必须通过 Shadow Mode。

---

# 42. 最小验证阶段

## Phase 0 — Capture

验证：

```text
Video
↔
Action
```

时间同步正确。

---

## Phase 1 — Tiny Overfit

数据：

```text
30~60 分钟
```

目标：

> 模型能够在训练数据上明显拟合玩家动作。

如果不能：

> Pipeline 有问题。

---

## Phase 2 — Offline Generalization

数据：

```text
3~5 小时
```

目标：

- 移动。
- 攻击。
- 闪避。
- 喝药。
- 简单交互。

---

## Phase 3 — Shadow Mode

实时预测但不控制。

---

## Phase 4 — Short Autopilot

AI 连续控制：

```text
3s
5s
10s
30s
```

逐步扩大。

---

## Phase 5 — Combat Loop

实现：

```text
普通敌人
+
精英
+
Boss
```

---

## Phase 6 — Exploration Loop

实现：

```text
跑图
+
采集
+
交互
+
简单寻路
```

---

## Phase 7 — Continuous Gameplay

目标：

> AI 从固定检查点 / 土地庙出发，自主运行 10~20 分钟。

过程中允许：

- 跑图。
- 探索。
- 采集。
- 小怪。
- Boss / 精英。
- 死亡。
- 重试。

但不能依赖显式世界状态机完成控制。

---

# 43. MVP 成功标准

MVP 不定义为：

> 打赢一个 Boss。

而定义为：

> AI 在固定游戏区域中，能够在仅使用视觉、最近动作历史和有限内部记忆的情况下，连续自主游玩。

建议初始成功标准：

```text
Autonomous Runtime >= 10 min

Manual Takeover Rate
持续下降

Stuck Ratio < 10%

能主动移动
能完成部分局部寻路
能进行采集 / 交互
能处理普通战斗
能主动闪避
能在低血状态治疗
能处理死亡并重新开始

关键闭环延迟：
P95 < 80ms
```

Boss 击杀率作为额外重要指标，但不是唯一成功条件。

---

# 44. 完整游戏的未来架构

对于完整类魂游戏，最终很可能需要双时间尺度：

```text
              Long-term Memory
                     │
                     ▼
                 Slow Policy
                  0.5~2Hz
                     │
               Context / Goal
                     │
                     ▼
Video History ─→ Fast Policy
                     ▲
                     │
              Action History
                     │
                     ▼
               Action Chunk
                     │
                     ▼
                Controller
```

Fast Policy：

- 战斗。
- 移动。
- 局部探索。
- 闪避。
- 镜头。
- 交互。

Slow Policy：

- 当前继续探索还是回退。
- 是否需要继续前进。
- 是否切换局部行为目标。
- 长时间没有进展时改变方向。

第一阶段不要求 Slow Policy。

---

# 45. 不建议第一版加入的内容

不要一开始加入：

```text
LLM
VLM Chat
OCR Pipeline
Object Detector
Game Knowledge Graph
Boss Skill Database
World State Machine
Rule Engine
A* Map
Quest System
```

除非实验明确证明某个信息瓶颈无法靠 Video Policy 学习。

原则：

> 能通过端到端学习解决，就先不显式建模。

但：

> 端到端不是宗教。

如果某个信息通过廉价稳定的结构化方式可直接获得，并且显著降低训练难度，则允许作为后续辅助输入。

---

# 46. 核心风险

## 46.1 Distribution Shift

模型小错误会导致进入训练集中没有的状态。

解决：

```text
Human Override
+
Correction Dataset
+
DAgger-style Iteration
```

---

## 46.2 Shortcut Learning

模型可能学习错误视觉相关性。

解决：

```text
Counter-example Eval
+
Failure Replay
```

---

## 46.3 时间标签错位

如果视频和输入错位：

> 模型会稳定学错。

因此同步优先级高于模型规模。

---

## 46.4 GPU 竞争

单 GPU 同时游戏和训练可能导致：

- 掉帧。
- Capture Delay。
- Dataset 污染。
- Input Delay。

默认：

```text
游戏运行
→
只采集

Episode 结束
→
训练
```

---

## 46.5 长程记忆不足

只靠 1~3 秒视频无法解决全部探索问题。

后续增加：

```text
Latent Memory
Slow Policy
Goal Conditioning
```

而不是立即回到大状态机。

---

# 47. 失败回退原则

系统如果连续出现：

```text
高置信错误动作
动作震荡
画面无进展
角色卡墙
输入异常
游戏失焦
模型超时
```

必须：

```text
Pause AI
→
Release Input
→
Human Override
```

不能尝试无限自恢复。

---

# 48. 工程模块建议

```text
ai_game_player/
├── capture/
│   ├── screen_capture
│   ├── input_capture
│   ├── audio_capture
│   └── clock
│
├── dataset/
│   ├── episode_store
│   ├── sample_builder
│   ├── replay_buffer
│   └── versioning
│
├── model/
│   ├── visual_encoder
│   ├── temporal_policy
│   ├── action_heads
│   └── checkpoint
│
├── train/
│   ├── trainer
│   ├── scheduler
│   ├── evaluator
│   └── registry
│
├── runtime/
│   ├── ring_buffer
│   ├── inference
│   ├── action_scheduler
│   ├── safety_filter
│   └── input_executor
│
├── evaluation/
│   ├── offline
│   ├── shadow
│   └── closed_loop
│
├── observability/
│   ├── metrics
│   ├── logs
│   └── dashboard
│
└── app/
    ├── observe_train
    └── autopilot
```

---

# 49. 两种模式最终定义

## OBSERVE_TRAIN

> 你玩，它看。

系统负责：

```text
Capture
Synchronize
Store
Build Samples
Train Candidate
Evaluate
```

---

## AUTOPILOT

> 它玩，你观察。

系统负责：

```text
Capture
Infer
Act
Monitor
```

出现错误：

```text
Human Override
→
Correction Recording
```

下一轮重新进入训练数据。

---

# 50. 一句话定义

> `AI Game Player` 是一个面向类魂和第三人称动作游戏的视觉端到端控制系统，通过同步观察玩家的连续游戏视频与操作行为，学习从 `Video History + Action History` 到 `Future Action Chunk` 的映射，并在真实游戏环境中闭环自主完成探索、移动、交互、采集和战斗。

---

# 51. 第一阶段真正要证明的事情

不是：

> AI 是否理解《黑神话：悟空》。

而是：

> 在不显式建立 Boss 技能表、地图状态机、任务状态机和世界模型的情况下，一个专用 Video-Action Policy 是否已经能够通过人类 demonstration 学会“玩”。

如果答案成立：

```text
复杂显式感知
+
人工规则
+
状态机
```

就不再是整个系统的核心。

系统核心会变成：

```text
高质量数据
+
时序模型
+
闭环训练
+
Recovery Data
+
评估体系
```

这就是本项目第一阶段的技术主线。
