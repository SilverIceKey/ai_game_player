# PROGRESS-20260813-autopilot-data-loop-v1

## 当前状态

- 当前主任务：AUTOPILOT 持续采集数据闭环（spec §26 修订）——代码完成，待游戏机实机验证
- 当前结论：
  - 接管不再切数据文件：AI 控制与人工接管连续记录同一条 episode，时间线事件全部走 telemetry JSONL（文件格式零改动）
  - 接管触发 = F12 toggle（保留）+ `safety.auto_takeover`（任何真实输入立即接管，默认开）；静默 `safety.resume_idle_ms`（默认 2500ms）自动恢复：清 pending chunk → 基于当前画面重新推理
  - 接管期间模型保持 Shadow Inference：proposed chunk 照常写 telemetry（`shadow=true`）但不下发；shadow 推理按 chunk 节奏限速，不 busy-loop
  - 每次推理（无论是否接管）落盘 `ai_proposed`：完整未截断 chunk + 延迟（inference/frame_age/queue_delay ms）+ gate/memory 诊断
  - marker 只有事实事件（EPISODE_START / HUMAN_OVERRIDE_START / AUTOPILOT_RESUME）；段语义由 `sample_builder` 构建期推导（采集层不判断训练价值）
  - Dataset Builder 段分类：human_demonstration / human_correction / autopilot_success / autopilot_failure；**PRE_OVERRIDE 段 anchor 整段剔除 + pre/override 段内 AI 动作一律不作 imitation target**；SessionDataset 构造时打印段分布与剔除计数

## 本轮改动

- `runtime/safety_filter.py`：`request_override()` / `request_resume()`（线程安全 flag，下一次 check_environment 生效，dead man switch 沿用）
- `app/autopilot.py`：`_correction_loop` → `_human_input_loop`（全程消费输入：触发接管 / correction 落盘 + 推入动作历史）；dispatch 循环加自动恢复；inference 循环 write_proposed + 仅 AI_CONTROL 提交 + 超时暂停仅 AI_CONTROL 生效；start 写 EPISODE_START
- `dataset/episode_store.py`：`write_marker` / `write_proposed`（marker 常量 MARKER_*）
- `dataset/sample_builder.py`：`build_samples(markers=, pre_override_window_us=, stats=)` + `_SegmentClassifier` + 每样本 `segment` 字段 + 失败段剔除
- `train/dataset.py`：telemetry marker 接入、段分布打印、旧数据（ai 动作无 marker）警告重新采集
- `config.py`：SafetyConfig 加 `auto_takeover` / `resume_idle_ms`；LabelsConfig 加 `pre_override_window_ms`（2000）；yaml 样例同步
- 计划：`docs/plans/PLAN-20260813-autopilot-data-loop-v1.md`

## 验证结果

- 已执行：`pytest` **325 passed**（新增：safety request_* 1 例、sample_builder 段分类/失败排除/无 marker 兼容 3 例、autopilot 自动接管+自动恢复+shadow 全链路 1 例）；compileall
- 未执行：真实键鼠触发接管延迟、真实游戏闭环、接管数据真实训练（需游戏机实机）
- 证据：`.venv/bin/python -m pytest -q` → 325 passed

## 风险与限制

- 接管期间 shadow 推理持续写 runtime memory deque（画面连续，判定合理）；实机若发现接管后行为异常可用 `reset_memory()` 验证
- 旧 AUTOPILOT 数据无 marker：AI 动作仍会全部当 target（构建期有警告），建议重新采集
- ai_proposed 体量：~10Hz × ~1KB/条 ≈ 36MB/h JSONL
- 输入 → 停止下发延迟 = poll 队列 + 一个 dispatch 周期（<10ms，实机待测）

## 下一步

1. 游戏机实机跑 AUTOPILOT：验证自动接管灵敏度（auto_takeover 误触/漏触）与 2500ms 恢复手感
2. 用新闭环数据训练一轮，看 human_correction 样本占比与 loss
3. 段标签可用于 spec §30 场景评估拆分（autopilot_success vs human_correction 分别看指标）
