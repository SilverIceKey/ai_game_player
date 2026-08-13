# ai_game_player

面向类魂 / 第三人称动作游戏的**端到端视觉控制 Agent**（SPEC v1.0）。

> 通过观察玩家真实游玩过程中的连续画面和同步操作记录，训练专用时序控制模型，
> 使其直接根据 `Video History + Action History`（+ 可选 `Audio History`，spec §8.5）
> 输出 `Future Action Chunk`，并在真实游戏环境中闭环运行。

权威规格：`docs/AI_Game_Player_SPEC_v1.0.md`（所有设计决策以它为准）。
当前重构计划：`docs/plans/PLAN-20260812-spec-v1-refactor-v1.md`。

## 两种顶层模式

| 模式 | 入口 | 说明 |
|---|---|---|
| `OBSERVE_TRAIN` | `python -m app.observe_train --game wukong` | 你玩，它看：截屏 + 输入采集 → 统一时钟同步 → Episode Store → 样本/训练。`--shadow` 开启 Shadow Mode（AI 实时预测但不执行，spec §41） |
| `AUTOPILOT` | `python -m app.autopilot --game wukong` | 它玩，你观察：加载 Active Model 闭环推理控制。`--dry-run` 不发真实输入 |

安全机制：F12 人工接管（spec §26）、失焦立即停手（§39）、Dead Man Switch（§40）、失败回退（§47）。

## 包结构（spec §48）

```text
capture/      屏幕采集（mss/WGC）+ 输入采集（键鼠 pynput / 手柄 XInput）+ 音频采集（WASAPI loopback，§8.5）+ 统一 monotonic 时钟
dataset/      Episode Store（§20）、样本构造（§22，含 action_label_offset_ms §12）、Replay Buffer（§28）、数据集版本（§29）
model/        Video-Action Policy 协议、占位 Policy、checkpoint 元数据（torch 实现后续引入）
train/        Trainer 骨架、训练时机调度（§6）、评估与 Model Registry（§7）
runtime/      Ring Buffer、Preprocess、Inference Worker、Action Scheduler（§15）、Safety Filter（§39）、输入执行器
evaluation/   离线评估（§35/§36）、Shadow（§41）、闭环指标（§37/§43）
observability/ 延迟分位统计（§32）、inference 日志（§33）
app/          observe_train / autopilot 两个 CLI 入口
config.py     YAML 配置加载校验
```

## 训练（spec §42 Phase 1：Tiny Overfit）

数据采集（OBSERVE_TRAIN）后，在游戏机上训练候选模型：

```bash
# 1. 安装 CUDA 版 PyTorch（训练与加载真实模型都需要；一次性）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2. 训练（产物为 Candidate，不会覆盖 Active Model，spec §7）
python -m app.train                          # 超参见 configs/settings.yaml 的 training 段
python -m app.train --epochs 20 --lr 0.0005  # 命令行覆盖

# 3. 用训练出的模型跑 AUTOPILOT（先 --dry-run 核对动作合理性）
python -m app.autopilot --game wukong --checkpoint checkpoints/model-v001 --dry-run
```

Phase 1 判据：train loss 显著下降 + 训练集按钮 P/R 明显高于随机。
做不到 → 按 spec §17 排查顺序检查（Timestamp → Labels → Dataset → 动作表示 → …）。

产物：`checkpoints/<model_version>/{model.pt, meta.json}`（含 dataset_version /
code_commit / training_config，spec §29 可复现）+ `checkpoints/registry.json`（§7 注册表）。
每个 epoch 结束都会更新一次 checkpoint（中断不丢进度）；用 `app.autopilot
--checkpoint checkpoints/model-vNNN` 指定版本加载。训练前会把样本引用的帧
一次解码缓存（内存够驻 RAM，不够自动落磁盘 memmap），不再每 epoch 重复解码视频。

### 音频模态（spec §8.5，可选，默认关闭）

游戏声音（Boss 抬手音效、受击音）可作为第三个输入模态。开启方式：`configs/settings.yaml`
设 `audio.enabled: true`（Windows 游戏机需 `pip install soundcard`，走 WASAPI loopback
录系统音频输出，不需要麦克风）。

**注意：开启前录制的 session 没有音频，不能直接用于带音频的训练，需重新采集**
（OBSERVE_TRAIN 重录；`app.train` 检测到无音频数据会明确报错）。带音频分支的
checkpoint 跑 AUTOPILOT 时也必须保持 `audio.enabled: true`，且 audio 参数与训练时一致。

## 开发与验证

```bash
.venv/bin/python -m pytest -q        # 全部单元测试（Linux 开发机可跑，平台依赖全延迟导入/注入）
python -m app.observe_train --help
python -m app.autopilot --help
```

平台约束：真实截屏/输入/手柄仅 Windows 实机可用（mss / windows-capture / pydirectinput / pynput / XInput），
Linux 开发机跑全部测试不触达真实设备。

## 文档结构

- `AGENTS.md` — 智能体协作与开发交接规则
- `docs/AI_Game_Player_SPEC_v1.0.md` — 项目权威规格
- `docs/agent-context.md` — 当前交接上下文（先读）
- `docs/plans/` / `docs/progress/` / `docs/reports/` — 计划 / 进度 / 报告
- `docs/archive/20260729-legacy-runtime/` — 已弃用的旧路线（ROI 感知 + FSM + LLM 复盘）文档归档
