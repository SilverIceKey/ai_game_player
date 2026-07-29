# ai_game_player

AI 游戏玩家项目，两条业务线：

- **全自动游戏**：面向原神、异环、黑神话悟空等 3D 游戏，由感知 + 决策 + 控制链路自动完成游戏行为
- **半自动 PK 决策**：面向幻唐志等回合制游戏，为玩家提供实时 PK 决策建议（人执行操作）

LLM 在本项目中的角色：**复盘分析与参数调整**，不直接参与端到端实时决策。

- 技术栈：Python
- 架构设计：见 `docs/plans/PLAN-20260729-project-skeleton-v1.md`

## 文档结构

- `AGENTS.md` — 智能体协作与开发交接规则
- `docs/agent-context.md` — 当前交接上下文（先读）
- `docs/guides/` — 系统说明、使用说明、架构说明
- `docs/plans/` — 阶段计划
- `docs/progress/` — 进度记录
- `docs/reports/` — 收口、联调、验收、专项报告
- `docs/test-data/` — 测试样本
- `docs/templates/` — 验收模板
- `docs/archive/` — 已归档文档
