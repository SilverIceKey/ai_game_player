# 交接上下文

## 当前状态

- 当前主任务：AUTOPILOT 持续采集数据闭环（spec §26 修订）——代码完成，待游戏机实机验证
- 当前阶段：Token Transformer 架构 + 数据闭环都已落地；**所有旧 checkpoint 作废（arch 标记缺失即 legacy/unsupported），旧 AUTOPILOT 数据无 marker 建议重采**
- 当前结论：
  - 项目方向：端到端 Video-Action Policy（唯一权威规格 `docs/AI_Game_Player_SPEC_v1.0.md`）
  - 旧路线已全部删除归档（`docs/archive/20260729-legacy-runtime/`），不做向前兼容
  - 三链路：`app.observe_train`（采集 + --shadow）、`app.train`（训练 + 依赖消融 + gate 统计 + 段分布）、`app.autopilot`（持续采集闭环：自动接管/自动恢复/shadow inference/proposed 落盘）
  - 架构：Token Transformer（ARCH_TAG=`token_transformer_v1`，唯一时序实现）；音频模态（§8.5）默认关
  - 数据闭环：同一条 episode 完整时间线（video + ai_proposed + executed + correction + marker）；段语义由 sample_builder 推导，失败段 AI 动作不回灌

## 本轮改动（2026-08-13 数据闭环）

- spec §26 重写（接管触发/自动恢复/持续采集/段分类规则）；计划 `docs/plans/PLAN-20260813-autopilot-data-loop-v1.md`；进度 `docs/progress/PROGRESS-20260813-autopilot-data-loop-v1.md`
- `safety_filter` 加编程模式切换；`autopilot` human_input_loop + shadow inference + marker；`episode_store` write_marker/write_proposed；`sample_builder` 段分类 + 失败剔除；config 加 auto_takeover/resume_idle_ms/pre_override_window_ms
- 测试 325 全绿（+5 新用例）

## 验证结果

- 已执行：`pytest` 325 passed、compileall、yaml 配置加载实测
- 未执行：游戏机实机接管手感/延迟、真实闭环数据训练（开发机无 GPU/无实机输入）
- 证据：`.venv/bin/python -m pytest -q` → 325 passed

## 风险与限制

- 接管期 shadow 推理持续写 runtime memory（画面连续判定合理，实机观察）；auto_takeover 可能误触（参数可关）
- 旧 AUTOPILOT 数据无 marker，AI 动作会全部当 target（构建期有警告）
- 训练/推理 memory 分布差、gate collapse 等上一轮风险不变

## 下一步

1. 游戏机实机 AUTOPILOT：验证自动接管灵敏度与恢复手感，采一轮闭环数据
2. `python -m app.train` 重新训练（旧 checkpoint 作废），看段分布 / dependency delta / gate 分布
3. 段标签接入场景评估拆分（evaluate_samples_by_scene 已备）

## 阅读顺序

1. `docs/AI_Game_Player_SPEC_v1.0.md`（§16 架构 + §26 数据闭环为最新约束）
2. `docs/plans/PLAN-20260813-autopilot-data-loop-v1.md`、`PLAN-20260813-token-transformer-v1.md`
3. `docs/progress/PROGRESS-20260813-autopilot-data-loop-v1.md`（最新进度）
4. 代码入口：`app/autopilot.py`（闭环）、`dataset/sample_builder.py`（段分类）、`model/torch_model.py`（架构）→ 契约 `capture/action.py`、`dataset/episode_store.py`、`config.py`
