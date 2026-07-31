# M3 计划：Runtime 对齐（技能化 / 仲裁 / 安全层 / Trace）

- 日期：2026-07-29
- 状态：**已确认**（用户授权：plan 直接通过，无需逐条确认）
- 上游：`docs/reports/REPORT-20260729-runtime-spec-review-v1.md`（说明书评审与取舍）、`docs/plans/PLAN-20260729-project-skeleton-v1.md`

## 1. 目标

把 M1 的"每 tick 重新决策 + 离散动作"主循环，重构为**技能生命周期 + 抢占 + 仲裁**的 Runtime 形态，同时补齐真机安全层。直接解决实机反馈的"动作僵硬、不顺畅"。

## 2. 范围

做：

1. **技能抽象** `core/skills/`：`GameSkill` 协议（name/priority/interruptible/can_start/start/tick/interrupt/dispose）+ `SkillTickResult`（RUNNING/SUCCEEDED/FAILED/NEEDS_REPLAN/NEEDS_HUMAN）
2. **两个技能实现**：
   - `ExplorationSkill`：包装 CoverageExplorer，生命周期内连续控制（move hold + 平滑转向），被"接敌"抢占
   - `CombatSkill`：包装 CombatDecision FSM，战斗闭环；脱战结束交还控制权
3. **动作仲裁** `core/control/arbiter.py`：优先级（人工急停 > 失焦保护 > 反射闪避 > 技能动作）、动作有效期（过期不执行）、仲裁决策写日志
4. **安全层** `core/safety.py`：
   - F12 全局急停（ctypes GetAsyncKeyState 轮询，无新依赖）：立即停止并释放全部按键，需人工确认恢复
   - 窗口失焦检测：失焦立即释放输入并暂停决策（复用 foreground 的窗口枚举）
   - 急停/失焦事件进日志与回放
5. **观测升级（轻量）**：GameState 透传 frame_id；感知读数带 confidence（有血条读数=高置信，隐藏降级）
6. **Trace**：主循环记录每 tick 感知/决策/执行延迟，session 结束输出 P50/P95 统计到 run 目录

不做（评审报告已论证）：Observation Store、Event Bus、Goal/Planner、VLM 导航、蒸馏、全套评估。

## 3. 主循环新形态

```text
每 tick：
  1. SafetyMonitor.check()        急停/失焦 → 立即释放输入 + 阻断本 tick 一切动作
  2. frame = source.grab()        frame_id 单调递增
  3. state = adapter.perceive()   带 confidence
  4. 事件判定：接敌/脱战/死亡 → Skill 抢占（CombatSkill ↔ ExplorationSkill）
  5. skill.tick(context) → 候选动作（含有效期）
  6. Arbiter.decide(候选动作, 安全状态) → 最终动作（过期/被抢占则丢弃）
  7. controller.execute() → recorder → 逐 tick 日志（原有格式 + skill 名）
  8. trace 延迟累计
```

## 4. 兼容性要求

- 现有行为不变的部分：HUD 感知（含动态血条）、WGC/mss 截屏、dry-run/calibrate/edit-roi/probe/review 全部 CLI 模式、逐 tick 三行日志格式（允许追加 skill 字段）、JSONL 回放
- 现有 120 个测试必须保持全绿（允许为适配接口而更新断言，但不得降低强度）
- 契约层修改仅限追加（GameState 加可选字段），不破坏既有 Protocol

## 5. 实现清单

| 文件 | 内容 |
| --- | --- |
| `core/skills/base.py` | GameSkill 协议 + SkillTickResult |
| `core/skills/exploration.py` | ExplorationSkill（包装 CoverageExplorer） |
| `core/skills/combat.py` | CombatSkill（包装 CombatDecision） |
| `core/control/arbiter.py` | ActionArbiter（优先级/有效期/急停阻断/仲裁日志） |
| `core/safety.py` | SafetyMonitor（F12 急停、失焦检测、释放输入） |
| `core/contracts.py` | GameState 追加 frame_id / confidence（可选字段） |
| `core/trace.py` | 延迟统计（P50/P95） |
| `apps/auto_player/main.py` | 主循环重构为技能调度 + 仲裁 + 安全检查 |
| `configs/wukong.yaml` / `settings.example.yaml` | safety 段（emergency_stop_key 等）、有效期参数 |
| `tests/` | 技能生命周期、抢占、仲裁优先级/过期、急停/失焦、trace 统计 |

平台约束不变：win32 API 全部 ctypes 延迟调用，Linux 测试全绿。

## 6. 验收

- `pytest` 全绿（新增：技能抢占、仲裁过期动作执行率=0、急停后无输入、失焦释放）
- 实机：动作连贯性改善（持续移动+平滑转向在技能生命周期内闭环）；F12 急停立即停手；切走窗口立即停止输入
