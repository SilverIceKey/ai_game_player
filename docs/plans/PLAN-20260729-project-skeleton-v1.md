# 项目骨架设计 v1

- 日期：2026-07-29
- 状态：**已确认**（2026-07-29 用户确认运行环境、首里程碑、合规边界、LLM 方案）

## 1. 业务范围

| 业务线 | 目标游戏 | 自动化程度 | 决策主体 |
| --- | --- | --- | --- |
| 全自动 | 黑神话悟空（首发）、原神、异环（3D） | 全自动 | 规则/状态机/行为树/RL 策略 |
| PK 辅助 | 幻唐志等回合制（场景不足，暂缓） | 半自动（建议，人执行） | 决策引擎出建议，人确认执行 |
| 复盘调参 | 全部 | 离线 | LLM（不进实时决策链路） |

核心约束：**LLM 不在端到端实时决策链路中**，只用于离线复盘与参数调优。

## 2. 已确认约束

- 运行环境：Windows PC 桌面端
- 合规边界：**仅截屏 + 模拟输入，不读内存、不注入**（保守路线）
- 硬件：RTX 2070 Super（8GB 显存），感知模型必须与游戏共存于同一显卡
  - 推论：实时感知只能用轻量模型（小型检测/分类网络、OCR、模板匹配），推理预算需严格控制
  - 推论：Ollama 本地 LLM 只跑离线复盘（游戏退出后或低负载时），不与游戏抢显存
- LLM 方案：默认本地 Ollama，Provider 接口兼容 Kimi / OpenAI

## 3. 分层设计

```text
┌─────────────────────────────────────────────┐
│ apps/            入口层：CLI / 建议展示 UI        │
├─────────────────────────────────────────────┤
│ games/           游戏适配层：每个游戏一个适配插件    │
│                  （状态定义、动作集、界面元素坐标）    │
├─────────────────────────────────────────────┤
│ core/            核心引擎层（与具体游戏无关）        │
│  ├─ perception/  感知：截屏、CV 识别、OCR          │
│  ├─ decision/    决策：状态机/行为树/规则引擎        │
│  ├─ control/     控制：键鼠/手柄输入模拟            │
│  └─ recorder/    记录：战斗日志、回放样本            │
├─────────────────────────────────────────────┤
│ llm/             LLM 离线服务：复盘分析、参数建议    │
├─────────────────────────────────────────────┤
│ configs/         参数与策略配置（LLM 调参的对象）    │
└─────────────────────────────────────────────┘
```

依赖方向只允许自上而下：`apps → games → core`，`llm` 只读 `recorder` 产出的日志、只写 `configs`。禁止反向依赖与循环依赖。

## 4. 目录结构

```text
ai_game_player/
├── apps/
│   ├── auto_player/      # 全自动入口（CLI）
│   └── pk_assistant/     # 回合制 PK 建议入口（暂缓，先留位）
├── core/
│   ├── perception/       # 截屏、模板匹配、OCR、轻量检测
│   ├── decision/         # 状态机 / 行为树 / 规则引擎
│   ├── control/          # 输入模拟（键鼠）
│   └── recorder/         # 战斗日志、状态快照、回放
├── games/
│   ├── wukong/           # 黑神话悟空适配（首发）
│   ├── genshin/          # 原神适配（后续）
│   └── ananta/           # 异环适配（待游戏上线）
├── llm/
│   ├── providers/        # Ollama / Kimi / OpenAI 兼容 Provider
│   ├── review/           # 复盘分析
│   └── tuning/           # 参数调整建议
├── configs/              # 游戏参数、策略配置（YAML）
├── tests/
└── docs/
```

## 5. 接口契约要点

- `GameAdapter`（games/ 每个插件实现）：
  - `perceive(frame) -> GameState`：把画面解析为本游戏的标准状态
  - `available_actions(state) -> list[Action]`：当前可执行动作集
  - `action_space() -> list[str]`：游戏动作定义（技能、移动、交互等）
- `DecisionEngine`（core/decision）：`decide(state) -> Action | Suggestion`
  - 全自动模式返回 `Action` 交给 control 执行
  - 半自动模式返回 `Suggestion` 交给 UI 展示
- `Controller`（core/control）：`execute(action) -> Result`，只依赖 Action 契约，不认识具体游戏
- `Recorder`（core/recorder）：全程记录 `state/action/result` 序列，产出回放样本，供 LLM 复盘
- `LLMProvider`（llm/providers）：统一 `complete(prompt) -> str`，Ollama 为默认实现，Kimi / OpenAI 走 OpenAI 兼容协议
- LLM 复盘（llm/review）：输入回放日志 → 输出问题诊断 + `configs/` 参数调整建议（人确认后生效）

## 6. 里程碑

- M1：core 骨架 + 黑神话悟空适配最小链路（截屏 → 状态识别 → 决策 → 模拟输入），首个场景在 M1 启动时与用户共同选定（候选：自动清小怪循环、自动跑图采集）
- M2：战斗日志录制 + LLM（Ollama 本地）复盘调参闭环
- M3：按复盘结论迭代黑神话策略与参数体系；评估原神适配启动
- M4+：原神 / 异环适配；幻唐志 PK 辅助（待场景充足）

## 7. 技术选型（M1 范围）

- 截屏：`mss`（Windows 桌面高性能截屏）
- 输入模拟：`pydirectinput`（DirectInput，游戏兼容性优于 pyautogui）
- CV：`opencv` + `numpy`；检测模型走 ONNX 推理，尺寸受 2070s 显存约束
- 配置：`pyyaml`
- LLM：`ollama` SDK + OpenAI 兼容协议（Kimi / OpenAI）
- 测试：`pytest`
