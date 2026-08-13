# PLAN: AUTOPILOT 持续采集数据闭环（2026-08-13）

## 需求（用户原话要点）

1. AI 控制与人工接管都连续记录**同一条 episode**：Video + AI Proposed Action + Actual Executed Action + Memory/Context + Timestamp/Latency。
2. 检测到真实键鼠/手柄输入 → 立即停止 AI 下发，保持模型 Shadow Inference，标记 `HUMAN_OVERRIDE_START`，切换为人工实际 Action 记录。
3. 连续 2~3s 无人工输入 → 清空 pending action chunk、基于当前画面重新推理、自动恢复，标记 `AUTOPILOT_RESUME`。
4. 不切分成独立数据文件；完整时间线上保留 `AI_CONTROL / PRE_OVERRIDE / HUMAN_OVERRIDE / POST_OVERRIDE` 语义。
5. 采集层只完整保存原始轨迹，不判断训练价值；Dataset Builder 区分 Human Demonstration / Human Correction / Autopilot Success / Autopilot Failure；**失败前的 AI Action 不得作为正确 imitation target 回灌**。

## 现状事实（已读代码确认）

- `capture/action.py`：ActionRecord.source ∈ {human, correction, ai}，定长 pack 进 actions.bin；追加新 source 不破坏旧文件（索引向后兼容），但 proposed chunk 是未来多步，不适合 actions.bin。
- `dataset/episode_store.py`：telemetry JSONL 已存在（`write_telemetry` / `reader.telemetry()`），调用方自定义字段——marker 与 proposed chunk 走这里，**不改任何文件格式**。
- `runtime/safety_filter.py`：模式机 AI_CONTROL⇄HUMAN_OVERRIDE，仅 F12 toggle 边沿触发；`dead_man_switch`（清队列+释放输入）已是现成语义。
- `app/autopilot.py`：dispatch 循环每轮 `check_environment` + `_track_mode`（已写 `human_override` telemetry）；`_correction_loop` 只在接管时记录 correction，AI_CONTROL 下输入被丢弃；inference 循环在 override 时仍 submit chunk（随后被 dead man switch 清掉，浪费且语义混乱）。
- `dataset/sample_builder.py`：targets 直接取动作流最近记录，**不区分 source**——当前 AUTOPILOT 数据的 AI 动作会无差别变成 imitation target（正是用户要修的问题）。
- `config.py`：SafetyConfig（override_key 等，game yaml）；LabelsConfig（action_label_offset_ms）。
- 推理侧 `policy.last_diagnostics`（fast_gate/memory_gate/memory_slots_filled）与 stats（inference_ms/frame_age_ms/queue_delay_ms）已具备，可直接入 telemetry。

## 设计

### 1. 时间线事件（采集层，只记录不判断）

全部写入现有 telemetry JSONL，两种新事件类型：

```json
{"type": "marker", "marker": "HUMAN_OVERRIDE_START" | "AUTOPILOT_RESUME" | "EPISODE_START", "timestamp_us": ...}
{"type": "ai_proposed", "timestamp_us": ..., "created_us": ..., "model_version": ...,
 "actions": [...每步 to_dict()...], "step_ms": 50.0,
 "inference_ms": ..., "frame_age_ms": ..., "queue_delay_ms": ...,
 "fast_gate": ..., "memory_gate": ..., "memory_slots_filled": ..., "shadow": true|false}
```

- marker 只有 START/RESUME 两种事实事件；`AI_CONTROL / PRE_OVERRIDE / HUMAN_OVERRIDE / POST_OVERRIDE` 是 Dataset Builder 从 marker 推导的**段标签**，采集层不写（避免采集层判断语义）。
- `ai_proposed` 每次推理都写（AI_CONTROL 下 shadow=false，override 下 shadow=true）——AI Proposed Action + Memory/Context（gates/slots）+ Latency 一并落盘。
- Actual Executed Action：现有 dispatch 循环的 `write_action(SOURCE_AI)`（过滤后动作）不变；人工动作 `SOURCE_CORRECTION` 不变。

`EpisodeStoreWriter` 加两个便捷方法（内部都是 write_telemetry）：
- `write_marker(marker: str, timestamp_us: int)`
- `write_proposed(chunk: ActionChunk, stats: dict, diagnostics: dict, shadow: bool)`

### 2. SafetyFilter：可编程模式切换

新增（线程安全，内部 flag 在下次 check_environment 生效，与 F12 toggle 并存）：
- `request_override()`：dispatch 循环检测到后进入 HUMAN_OVERRIDE（dead man switch 立即清队列+释放输入，沿用现有路径）。
- `request_resume()`：回到 AI_CONTROL。

F12 手动 toggle 保留（显式急停语义不变）。

### 3. autopilot 行为改造（app/autopilot.py）

- `_correction_loop` → `_human_input_loop`：任何输入事件都（a）更新 `_last_human_input_us`；（b）AI_CONTROL 下收到输入 → `safety.request_override()`；（c）HUMAN_OVERRIDE 下照旧写 SOURCE_CORRECTION 记录，**并推入 `_action_history`**（shadow 推理的动作上下文连续）。
- `_dispatch_loop`：每轮检查——HUMAN_OVERRIDE 且 `now - _last_human_input_us > safety.resume_idle_ms` → `scheduler.clear()`（清 pending chunk）+ `safety.request_resume()`。不清 memory（接管前后是同一场景连续画面，不属 §8.3 Hard Reset 清单）。
- `_track_mode`：迁移时写 marker——进入 override 写 `HUMAN_OVERRIDE_START`，回到 AI_CONTROL 写 `AUTOPILOT_RESUME`（替换现有 `human_override` telemetry，旧事件名弃用）。
- `_inference_loop`：每次推理都 `writer.write_proposed(...)`；仅当 `safety.mode == AI_CONTROL` 才 `submit_chunk`（override 期间 = shadow inference，不下发）；推理超时暂停逻辑只在 AI_CONTROL 下生效（override 时 AI 本就不控制，不报警）。恢复后 has_chunk=false → 下一轮循环自动基于最新画面重新推理（天然满足"重新推理"）。
- start() 时写 `EPISODE_START` marker。

### 4. Dataset Builder 分类（dataset/sample_builder.py）

`build_samples(..., markers: list[dict] | None = None, pre_override_window_us: int = 2_000_000)`：

- 无 markers（OBSERVE_TRAIN 数据）→ segment 全部 `human_demonstration`，行为与现在完全一致。
- 有 markers：推导段——每对 (HUMAN_OVERRIDE_START, AUTOPILOT_RESUME) 之间为 override 段；START 前 pre_override_window 为 pre 段；RESUME 后等长窗口为 post 段；其余 AI_CONTROL。
- 每个样本标注 `segment` ∈ {human_demonstration, human_correction, autopilot_success, autopilot_failure}：
  - anchor 落在 override 段 → `human_correction`（target 是 correction 记录）
  - anchor 落在 pre 段 → `autopilot_failure`
  - AUTOPILOT 数据其余 → `autopilot_success`
- target 过滤规则：source ∈ {human, correction} 恒可用；source == ai 仅在样本 segment == autopilot_success 时可用。样本任一 target 槽不可用 → 跳过该 anchor（与现有跳过规则一致），并统计跳过数。
- 效果：失败前（pre 段）的 AI 动作不会成为 imitation target；correction 段用人类动作做 target；成功的 AI 段可自模仿。

`SessionDataset`：`reader.telemetry()` 取 marker 事件传给 build_samples；构造后打印 segment 分布统计（每类样本数 + 跳过数）——采集不判断、构建期可见。

### 5. 配置

- `SafetyConfig`（game yaml）：`auto_takeover: bool = True`、`resume_idle_ms: float = 2500.0`（用户说的 2~3s，取中值，配置化）。
- `LabelsConfig`（settings.yaml）：`pre_override_window_ms: float = 2000.0`（post 段复用同值）。
- settings.example.yaml / wukong.yaml 同步注释样例。

### 6. 不改的东西

- 文件格式（actions.bin / frames.idx / telemetry.jsonl 结构不变，只多两种 telemetry 事件类型）。
- OBSERVE_TRAIN 链路（无 markers → 行为完全不变）。
- 执行器、截屏、训练器、模型。
- 旧的 F12 toggle、失焦保护、§47 超时暂停语义。

## 影响分析

- 调用链变化：input_capture.poll 从"仅接管期消费"变为"全程消费并可能触发接管"——dispatch 循环每 2ms 轮询 safety，输入→停止下发的延迟 = poll 队列延迟 + 一个 dispatch 周期（<10ms）。
- 旧 AUTOPILOT 数据（无 marker telemetry）：按无 markers 处理 = human_demonstration 规则但含 ai source 动作——**与现状一致地全部当 target**。判定：旧数据本就质量不明，且用户要求不向前兼容，不迁移；新采集才有闭环语义。在 sample_builder 中对"ai source 动作但无 markers"的情况打印一次警告提示重新采集。
- telemetry 体积：ai_proposed 每次推理一条（~10Hz × 每条约 1KB），一小时约 36MB JSONL，可接受。

## 修改文件清单

1. `dataset/episode_store.py`：write_marker / write_proposed（+ reader 无改动，telemetry() 已有）
2. `runtime/safety_filter.py`：request_override / request_resume（锁保护 flag，check_environment 消费）
3. `app/autopilot.py`：human_input_loop / dispatch 自动恢复 / inference shadow + write_proposed / marker 写入
4. `dataset/sample_builder.py`：markers 参数 + 段推导 + segment 标注 + target 过滤 + 跳过统计
5. `train/dataset.py`：telemetry → markers 传入、segment 分布打印、无 markers 含 ai 动作的警告
6. `config.py` + `configs/settings.example.yaml` + `configs/wukong.yaml`：新配置项
7. `capture/action.py`：无改动（source 三值够用）
8. 测试：`tests/test_app_autopilot.py`（接管/恢复/shadow 全链路）、`tests/test_sample_builder.py`（段分类与失败排除）、`tests/test_safety_filter.py`（request_*）、`tests/test_train_dataset.py` 视断言调整
9. 文档：本计划复制到 `docs/plans/PLAN-20260813-autopilot-data-loop-v1.md`；spec §26/§27 相关段落同步；完成后 `docs/progress/PROGRESS-20260813-autopilot-data-loop-v1.md` + `docs/agent-context.md`

## 验证

- 单元/集成测试全绿（现有 320 + 新增）
- 关键用例：AI_CONTROL 中注入人工输入 → marker HUMAN_OVERRIDE_START + 停止 submit + shadow proposed 落盘；静默 resume_idle_ms 后 → scheduler 清空证据 + AUTOPILOT_RESUME + 恢复 submit；sample_builder 对构造的 marker 序列：pre 段 ai target 被排除、correction 段 target 用人类动作
- 不验证（无实机）：真实键鼠触发延迟、真实游戏闭环——开发机不可验，明确标注

## 风险

- override 期间 shadow 推理持续写 memory deque：恢复时 memory 含接管期信息——判定为合理（画面连续），若实机发现污染再议（reset_memory API 已存在）。
- input_capture 队列若积压旧事件，takeover 触发会用到旧时间戳：request_override 用 now_us() 而非事件时间戳，避免滞后。
