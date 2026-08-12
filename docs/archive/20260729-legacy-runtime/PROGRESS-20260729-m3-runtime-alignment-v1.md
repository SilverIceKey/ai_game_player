> **已归档**（2026-08-12）：本文档属于已弃用的「ROI 感知 + FSM 决策 + LLM 复盘」旧路线，不再适用。
> 替代文档：`docs/AI_Game_Player_SPEC_v1.0.md` 与 `docs/plans/PLAN-20260812-spec-v1-refactor-v1.md`。

# 进度：M3 Runtime 对齐（技能化 / 仲裁 / 安全层 / Trace）

- 日期：2026-07-29
- 阶段：M3 代码实现完成，待实机验证动作连贯性与安全层

## 当前状态

- 依据 `docs/reports/REPORT-20260729-runtime-spec-review-v1.md` 的取舍，执行 `docs/plans/PLAN-20260729-m3-runtime-alignment-v1.md`，全部落地
- 单元测试 144 passed（120 → 144，新增 24 例）；说明书原文存 `docs/reference/`
- 主循环重构为：安全检查 → 截屏(frame_id) → 感知(带置信度) → 技能调度（抢占）→ 仲裁（优先级+TTL）→ 执行 → 记录/日志 → trace

## 最近关键结论

- 技能化：GameSkill 协议 + 五态 SkillTickResult；ExplorationSkill/CombatSkill 分别包装现有 CoverageExplorer/CombatDecision（内部逻辑不动）；接敌抢占、脱战断点经共享黑板移交恢复
- 仲裁：ActionArbiter 优先级 EMERGENCY > FOCUS_GUARD > REFLEX > SKILL，动作 TTL（默认 500ms）过期丢弃，全量仲裁日志；REFLEX 级本期无生产者（预留）
- 安全层：F12 急停（边沿检测 toggle，急停中每 tick 释放全部按键+仲裁阻断，再按 F12 恢复）；失焦立即释放输入；NEEDS_HUMAN（死亡）闩锁不自动恢复
- Trace：每 tick 三段延迟，session 结束写 `runs/<ts>/trace_summary.txt`（P50/P95/max）
- 兼容性：dry-run/calibrate/edit-roi/probe/review 全部保留；逐 tick 日志 intent 行追加 skill= 字段，原格式不变

## 下一步动作

- 用户 Windows 实机：`git pull` → 正式跑一段，对比动作连贯性（M3 核心目标）
- 验证 F12 急停、切窗口失焦停手；看 `trace_summary.txt` 的 P95 是否逼近 100ms 帧预算
- 实机确认后补 M1/M3 验收报告

## 阻塞项

- 无

## 未证实风险

- GetAsyncKeyState 在独占全屏/权限差异下的稳定性；失焦检测在 WGC 后台截屏场景下用户切窗看日志会停手（设计行为，实机确认不误伤）
- 2070s 上 perceive（截屏+CV）能否撑住 10fps 未实测
- 仲裁日志每 tick 一行，长会话日志体积大（可降 DEBUG，本期保留 INFO 便于联调）

## 验证

- 已执行：`pytest` 144 passed（技能五态/抢占/断点移交/NEEDS_HUMAN 闩锁/仲裁优先级/过期/TTL/急停阻断/失焦释放/trace 百分位/主循环急停集成）；`compileall` 通过
- 未执行：实机动作连贯性、急停/失焦实机行为、trace 实机数据
