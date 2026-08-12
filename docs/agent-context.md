# 交接上下文

## 当前状态

- 当前主任务：SPEC v1.0 全量重构（本轮代码落地完成）
- 当前阶段：待 Windows 实机验证（spec §42 Phase 0：Video ↔ Action 同步误差实测 <10ms）
- 当前结论：
  - 项目方向已整体切换：端到端 Video-Action Policy（唯一权威规格 `docs/AI_Game_Player_SPEC_v1.0.md`）
  - 旧路线（ROI 感知 + FSM 决策 + LLM 复盘）已全部删除，文档归档 `docs/archive/20260729-legacy-runtime/`，不做向前兼容
  - 新骨架按 spec §48 落成：OBSERVE_TRAIN / AUTOPILOT 双模式链路全通（模型本体为占位 Policy，未引入 torch）

## 本轮改动

- 新增包：`capture/`（统一时钟 §11、NormalizedAction §9、mss/WGC 截屏、pynput 键鼠 + XInput 手柄采集）、`dataset/`（Episode Store §20、样本构造 §22/§12、Replay Buffer §28、版本 §29）、`runtime/`（§30 线程组件、Safety Filter §39/§40/§26/§47、键鼠执行器）、`model/ train/ evaluation/ observability/`（协议与骨架）、`app/`（observe_train / autopilot 两个 CLI）
- 删除：`llm/ apps/ games/ core/` 全部 + 22 个旧测试 + `.env.example`
- 配置：`config.py` 新 schema；`configs/settings.example.yaml` 与 `configs/wukong.yaml` 重写；pyproject 新包名 + pynput
- 修复：`dataset/episode_store.py` 帧索引/动作流改逐条 flush（防崩溃丢同步数据）
- 文档：README 重写、计划 `docs/plans/PLAN-20260812-spec-v1-refactor-v1.md`、进度 `docs/progress/PROGRESS-20260812-spec-v1-refactor-v1.md`

## 验证结果

- 已执行：`pytest` 270 passed、compileall、两个 CLI `--help` 正常
- 未执行：Windows 实机（截屏/输入/手柄/同步误差/SHADOW 全部待实测）
- 证据：`.venv/bin/python -m pytest tests/ -q` → 270 passed

## 风险与限制

- 同步误差 <10ms（§11）未实测——实机第一验证项
- pynput 全屏捕获率、鼠标差分 vs raw input、XInput 摇杆方向约定、60fps cv2 写入开销均未实测
- wukong.yaml 中标注"实机校准"的键位与 executor.pixels_per_unit 必须实机核对
- 手柄输出（ViGEm）、torch 训练、卡墙/震荡检测为接口预留未实现

## 下一步

1. Windows 实机：`python -m app.observe_train --game wukong` 采集首段数据，实测同步误差（Phase 0）
2. 实机校准键位后试 `app.autopilot --dry-run` 与 `--shadow`
3. 数据 30~60 分钟后引入 torch 实现 model/train（Phase 1 tiny overfit）

## 阅读顺序

1. `docs/AI_Game_Player_SPEC_v1.0.md`（唯一权威规格）
2. `docs/plans/PLAN-20260812-spec-v1-refactor-v1.md`（本轮重构计划：复用/弃用清单与关键设计决策）
3. `docs/progress/PROGRESS-20260812-spec-v1-refactor-v1.md`（最新进度）
4. 代码入口：`app/observe_train.py`、`app/autopilot.py` → 契约 `capture/action.py`、`config.py`
