# PROGRESS-20260812-spec-v1-refactor-v1：SPEC v1.0 全量重构进度

> 计划：`docs/plans/PLAN-20260812-spec-v1-refactor-v1.md`；规格：`docs/AI_Game_Player_SPEC_v1.0.md`

## 当前状态

- 当前主任务：SPEC v1.0 全量重构（本轮已完成代码落地）
- 当前阶段：代码与文档收口完成，待 Windows 实机验证（spec §42 Phase 0：Video ↔ Action 时间同步）
- 当前结论：旧路线（ROI 感知 + FSM + LLM）已整体删除；新骨架按 spec §48 落成，OBSERVE_TRAIN 与 AUTOPILOT 双模式链路全通（占位 Policy）

## 本轮改动

- 删除弃用：`llm/ apps/ games/ core/ `全部（含 decision/navigation/skills/arbiter/ROI 感知/旧 recorder）+ 22 个旧测试 + `.env.example`
- 新增 `capture/`：clock（§11 统一时钟）、action（§9 NormalizedAction/ActionRecord/ActionChunk）、screen（mss/WGC 迁移改造，帧级时间戳）、input（pynput 键鼠 + XInput 手柄采集）
- 新增 `dataset/`：episode_store（§20 session 结构）、episode（§21 手动切分）、sample_builder（§22 + §12 offset）、replay_buffer（§28）、versioning（§29）
- 新增 `runtime/`：ring_buffer、preprocess（§14）、inference（§33）、action_scheduler（§15）、safety_filter（§39/§40/§26/§47）、input_executor、gamepad_executor（ViGEm 占位）、null_executor
- 新增 `model/ train/ evaluation/ observability/`：Policy 协议 + Placeholder/Random、checkpoint（§29）、Trainer 骨架（无 torch 明确报错）、Registry（§7）、离线/Shadow/闭环指标（§35-§37/§41）、延迟分位与 §33 日志
- 新增 `app/`：`python -m app.observe_train`（含 --shadow）、`python -m app.autopilot`（含 --dry-run，§30 线程结构，F12 接管记 correction）
- 顶层 `config.py` 新 schema；`configs/settings.example.yaml` + `configs/wukong.yaml` 重写；`pyproject.toml` 新包名 + pynput（删 ruamel/llm extra）
- 修 bug：`dataset/episode_store.py` frames.idx/actions.bin 改为逐条 flush（缓冲落盘会导致崩溃时同步索引落后于视频）
- 文档：README 重写、agent-context 更新、旧路线 10 份文档归档 `docs/archive/20260729-legacy-runtime/`、`.vscode/launch.json` 换两个新入口、`.gitignore` 加 settings.yaml/sessions/

## 验证结果

- 已执行：`pytest` **270 passed**；`compileall` 通过；`python -m app.observe_train --help` / `app.autopilot --help` 正常；`test_config_loader` 锁定两份 YAML 与 schema 一致
- 未执行：Windows 实机验证（真实截屏/输入/手柄/同步误差/SHADOW）
- 证据：`.venv/bin/python -m pytest tests/ -q` → 270 passed in 3.35s

## 风险与限制

- 采集同步误差 <10ms（§11）未实测，实机第一件事就是验证 Phase 0
- pynput 钩子在独占全屏下捕获率、鼠标差分 vs raw input 偏差、XInput 摇杆方向约定未实测
- 60fps cv2 视频写入 CPU 开销未实测（备选：降 source_fps）
- mp4v 有损编码，训练帧有压缩噪声（如需无损改帧序列存储，后续决策）
- 手柄输出（ViGEm）、torch 模型与真实训练、卡墙/震荡检测（§47 两条）为接口预留

## 下一步

1. Windows 实机跑 OBSERVE_TRAIN，验证 Phase 0（同步误差实测 + 首段数据采集）
2. 实机核对 wukong.yaml 键位（标注"实机校准"项）与 executor.pixels_per_unit
3. 数据量到 30~60 分钟后引入 torch，实现 model/ + train/（spec §16-§19，Phase 1 tiny overfit）
