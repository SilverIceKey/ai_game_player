# ai_game_player

面向类魂 / 第三人称动作游戏的**端到端视觉控制 Agent**（SPEC v1.0）。

> 通过观察玩家真实游玩过程中的连续画面和同步操作记录，训练专用时序控制模型，
> 使其直接根据 `Video History + Action History` 输出 `Future Action Chunk`，
> 并在真实游戏环境中闭环运行。

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
capture/      屏幕采集（mss/WGC）+ 输入采集（键鼠 pynput / 手柄 XInput）+ 统一 monotonic 时钟
dataset/      Episode Store（§20）、样本构造（§22，含 action_label_offset_ms §12）、Replay Buffer（§28）、数据集版本（§29）
model/        Video-Action Policy 协议、占位 Policy、checkpoint 元数据（torch 实现后续引入）
train/        Trainer 骨架、训练时机调度（§6）、评估与 Model Registry（§7）
runtime/      Ring Buffer、Preprocess、Inference Worker、Action Scheduler（§15）、Safety Filter（§39）、输入执行器
evaluation/   离线评估（§35/§36）、Shadow（§41）、闭环指标（§37/§43）
observability/ 延迟分位统计（§32）、inference 日志（§33）
app/          observe_train / autopilot 两个 CLI 入口
config.py     YAML 配置加载校验
```

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
