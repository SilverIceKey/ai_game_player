# PLAN-20260812-spec-v1-refactor-v1：按 AI Game Player SPEC v1.0 全量重构

> 依据：`docs/AI_Game_Player_SPEC_v1.0.md`（唯一权威来源，下文 §N 均指该 spec 章节）
> 范围决策（用户 2026-08-12 确认）：全量骨架含 AUTOPILOT 闭环；按 §48 新包结构；键鼠 + 手柄；不做向前兼容。

## 1. 目标与边界

将现有「ROI 感知 + FSM 决策 + 栅格导航 + LLM 复盘」项目整体重构为 spec 定义的端到端视觉控制系统：

```text
Video History + Action History (+ Optional Memory/Goal)
        ↓
   Video-Action Policy
        ↓
 Future Action Chunk
```

本轮交付（对应 §42 Phase 0 可验证 + 全链路骨架）：

- `OBSERVE_TRAIN`：截屏 + 输入采集 → 统一时钟同步 → Episode Store → 样本构造 → Replay Buffer →（训练接口骨架）
- `AUTOPILOT`：§30 线程结构闭环（Capture → Ring Buffer → Preprocess → Inference → Action Scheduler → Safety Filter → Input Executor），模型本体用占位 Policy
- SHADOW 子状态（§41）、Human Override（§26）、Dead Man Switch（§40）、Safety Filter（§39）

明确不做（本轮）：PyTorch 模型实现与真实训练、手柄输出（ViGEm）、自动 Episode 检测、Slow Policy/Memory/Goal Conditioning（§44）、任何旧功能兼容层。

## 2. 复用 / 弃用清单

### 2.1 迁移复用

| 现有文件 | 新位置 | 改造点 |
|---|---|---|
| `core/perception/mss_source.py` | `capture/screen/mss_source.py` | `grab()` 返回 `(frame, timestamp_us)`，抓帧时刻打统一时钟 |
| `core/perception/wgc_source.py` | `capture/screen/wgc_source.py` | 同上（帧回调内打点） |
| `core/perception/source_factory.py` | `capture/screen/source_factory.py` | 适配新 config 类型 |
| `core/perception/foreground.py` | `capture/screen/foreground.py` | 原样迁移 |
| `core/safety.py` | `runtime/safety_filter.py` | 扩展为 §39 Safety Filter + §40 Dead Man Switch + §26 Human Override |
| `core/control/directinput.py` | `runtime/input_executor.py` | 改为 NormalizedAction 驱动：任意键 hold/release、鼠标连续移动、鼠标按键 |
| `core/control/null_controller.py` | `runtime/null_executor.py` | 适配新契约 |
| `core/trace.py` | `observability/metrics.py` | 扩展 P50/P90/P95/P99/MAX（§32） |
| `core/config.py` YAML 校验工具 | 顶层 `config.py` | 新 schema，工具函数复用 |

### 2.2 直接弃用删除

- `llm/`（§45 禁 LLM/VLM）
- `apps/pk_assistant/`（回合制游戏超出 §1 边界）
- `core/decision/`、`core/navigation/`、`core/skills/`（状态机/规则/地图，§45 禁止）
- `core/control/arbiter.py`（被 action_scheduler + safety_filter 取代）
- `core/perception/` 的 `bars.py regions.py odometry.py walkable.py base.py`（显式 ROI 感知，§16 明确不要求）
- `games/`（全部规则适配器）
- `apps/auto_player/` 全部（被 `app/` 取代；ROI 校准工具随 ROI 感知弃用）
- `core/recorder/`（被 §20 Episode Store 取代）
- `core/contracts.py`（被 `capture/action.py` NormalizedAction 契约取代）
- 上述模块对应的旧测试、`configs/settings.example.yaml`、`configs/wukong.yaml`（重写）

## 3. 新包结构（§48）

```text
capture/
├── clock.py              # 统一 monotonic clock now_us()（§11）
├── action.py             # NormalizedAction / ActionChunk 契约（§9）
├── screen/               # mss_source / wgc_source / source_factory / foreground（迁移改造）
└── input/
    ├── base.py           # InputCapture 协议：事件 (timestamp_us, NormalizedAction)
    ├── keyboard_mouse.py # pynput 全局监听（延迟导入）
    └── gamepad.py        # XInput via ctypes（零新依赖）

dataset/
├── episode_store.py      # §20 sessions/<id>/{manifest.json, video/, frames.idx, actions.bin, events.jsonl, telemetry.jsonl}
├── episode.py            # 手动 START/STOP EPISODE（§21）
├── sample_builder.py     # §22 样本构造；action_label_offset_ms 可配（§12）
├── replay_buffer.py      # §28 四类加权采样
└── versioning.py         # §29 dataset-vNNN

model/
├── policy.py             # VideoActionPolicy 协议 + PlaceholderPolicy + RandomPolicy
├── visual_encoder.py     # 接口定义（§18 训练策略注明，torch 后续引入）
├── temporal_policy.py    # 接口定义
├── action_heads.py       # 接口定义（§19 拆 Head）
└── checkpoint.py         # §29 模型版本元数据

train/
├── trainer.py            # 骨架：无 torch 时明确报错指引（§6）
├── scheduler.py          # 训练时机：仅 episode 结束/暂停（§6）
├── evaluator.py          # 离线评估入口
└── registry.py           # candidate → evaluate → promote/reject（§7），切换只在 episode 边界

runtime/
├── ring_buffer.py        # 定长历史帧窗口
├── preprocess.py         # resize 384×216（§14）+ 归一化
├── inference.py          # Inference Worker（§33 日志字段）
├── action_scheduler.py   # 50ms step 派发 chunk（§15）
├── safety_filter.py      # §39 + §40 + §26
├── input_executor.py     # NormalizedAction → 键鼠
├── gamepad_executor.py   # 手柄输出接口（ViGEm 占位，明确报错）
└── null_executor.py      # dry-run

evaluation/
├── offline.py            # §35/§36 eval set 约定 + Movement/Camera Error、按钮 P/R
├── shadow.py             # §41 AI 意图 vs 玩家操作对齐指标
└── closed_loop.py        # §37 闭环指标骨架

observability/
├── metrics.py            # §32 延迟分位 + capture_fps/dropped/queue_delay
└── logs.py               # §33 inference JSONL 日志

app/
├── observe_train.py      # OBSERVE_TRAIN 入口（--shadow 可开 §41）
└── autopilot.py          # AUTOPILOT 入口（§30 线程结构）

config.py                 # 新 YAML schema 加载校验
```

## 4. 关键设计决策

1. **时间同步（§11，最高优先级）**：`capture/clock.py` 唯一时钟源 `now_us()`；帧在 grab 回调打点、输入事件在回调打点；目标 <10ms。
2. **动作空间与键位解耦（§9）**：`NormalizedAction`（move_x/y、camera_x/y 连续 [-1,1] + 布尔按钮集）；键位映射由 config `keys:` 驱动。
3. **手柄**：采集用 XInput ctypes 轮询（零新依赖）；输出需 ViGEm 内核驱动，本轮只立接口 + 明确报错。
4. **键鼠采集**：新增依赖 `pynput`（纯 Python、成熟；延迟导入）。鼠标 delta 用移动事件差分近似（raw input 精确采集留后续）。
5. **模型本体**：本轮不引入 torch。PlaceholderPolicy/RandomPolicy 支撑 AUTOPILOT 全链路；`train/trainer.py` 在无 torch 时明确报错。loss 权重、采样权重全部配置化（§23/§28）。
6. **平台约束沿用老规矩**：win32/pynput/pydirectinput/XInput 全部延迟导入 + 依赖注入，Linux 开发机跑全部单元测试。
7. **线程结构（§30）**：capture / inference / input 分线程，queue 解耦，禁止单线程阻塞。
8. **失败回退（§47）**：本轮实现失焦/模型超时/急停三条回退路径；震荡与卡墙检测留接口。

## 5. 配置新 schema

```yaml
window: {title, capture_backend, foreground_on_start}
capture: {source_fps: 60}
model: {sample_fps: 12, history_frames: 16, input_width: 384, input_height: 216}
prediction: {action_step_ms: 50, future_action_steps: 4}
labels: {action_label_offset_ms: 0}        # §12 实验搜索 0~250
input_device: keyboard_mouse | gamepad
keys: {...}                                 # §9 全动作 → 实际键位
safety: {override_key: F12, episode_key: F9, stop_on_focus_lost, max_button_hold_ms, max_camera_delta, max_action_rate_hz}
sampling: {historical: 0.5, recent: 0.25, correction: 0.2, rare: 0.05}
loss_weights: {move: 1.0, camera: 1.0, button: 1.0, temporal: 1.0}
```

## 6. 实施步骤

1. 本计划落盘（已完成）
2. 删除弃用代码与旧测试、旧 configs
3. `capture/`：clock、action 契约、screen 迁移、input 采集
4. `dataset/`
5. `runtime/`
6. `model/` + `train/` + `evaluation/` + `observability/`
7. `app/` 两个 CLI 入口
8. `config.py` + configs 重写 + `pyproject.toml`（新包名、删 llm extra、加 pynput）
9. 重写 tests（全依赖注入，不触达 win32）
10. 验证：pytest 全绿 + compileall + CLI `--help` 检查
11. 文档收口：README 重写、agent-context 更新、旧 plans/progress/guides 归档

## 7. 验证与风险

- 本轮可验证：pytest（Linux 全依赖注入）、compileall、CLI 启动检查
- 需 Windows 实机（移交用户）：采集同步误差实测（§11 <10ms）、手柄 XInput、AUTOPILOT 真实输入、SHADOW 实机
- 风险：pynput 钩子在独占全屏下捕获率未实测；鼠标差分 vs raw input 偏差未实测；60fps cv2 视频写入 CPU 开销未实测（备选：降 source_fps 或帧序列存储）
