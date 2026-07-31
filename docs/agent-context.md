# 交接上下文

## 当前状态

- 当前主任务：M3——Runtime 对齐（代码完成，待实机验证）
- 当前阶段：Windows 实机验证（动作连贯性、F12 急停、失焦停手、trace）
- 当前结论：
  - 说明书评审落盘 `docs/reports/REPORT-20260729-runtime-spec-review-v1.md`：控制面思想吸收，多模型矩阵/蒸馏不做
  - M3 全部落地，`pytest` 144 passed；主循环已重构为 安全→感知→技能调度（抢占）→仲裁→执行→trace
  - 说明书原文：`docs/reference/通用视觉游戏 Agent Runtime 技术规格说明书.docx`

## 本轮改动

- 新增 `core/skills/`（base 协议五态、exploration/combat 包装技能、scheduler 抢占调度）
- 新增 `core/control/arbiter.py`（优先级+TTL+仲裁日志）、`core/safety.py`（F12 急停 toggle、失焦释放）、`core/trace.py`（延迟 P50/P95）
- 修改 `apps/auto_player/main.py`（主循环重构，日志 intent 行追加 skill=）、`core/contracts.py`（GameState 加 frame_id/confidence 可选字段）、`games/wukong/combat.py`（断点移交只读接口）、`games/wukong/adapter.py`（safety 配置 + confidence 透传）、`configs/wukong.yaml`（safety 段 + action_ttl_ms）
- 新增测试 24 例（skills/arbiter/safety/trace）
- 文档：评审报告 + M3 计划 + M3 进度

## 验证结果

- 已执行：`pytest` 144 passed、`compileall` 通过
- 未执行：实机动作连贯性、急停/失焦实机行为、trace 实机数据
- 证据：`.venv/bin/python -m pytest -q` → 144 passed in 1.31s

## 风险与限制

- 急停轮询在独占全屏/权限差异下稳定性未知；失焦停手是设计行为（WGC 后台截屏时切窗看日志会停手），实机确认
- perceive 延迟能否撑住 10fps 未实测（看 trace_summary.txt）
- 仲裁日志 INFO 级每 tick 一行，长会话体积大
- settings.yaml 为本地文件：pull 后需对照 example 补新字段（老问题，本轮 wukong.yaml 新增 safety 段在 git 内会自动更新）

## 下一步

- 用户实机验证 M3 → 补验收报告（docs/reports/）
- 后续可选：VLM 低频导航实验、Event Bus/Observation Store（M3 实机结论后再评估）
- MP 感知已就绪未接决策；法术进决策排期未定
