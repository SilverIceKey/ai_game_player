> **已归档**（2026-08-12）：本文档属于已弃用的「ROI 感知 + FSM 决策 + LLM 复盘」旧路线，不再适用。
> 替代文档：`docs/AI_Game_Player_SPEC_v1.0.md` 与 `docs/plans/PLAN-20260812-spec-v1-refactor-v1.md`。

# 进度：项目骨架落地

- 日期：2026-07-29
- 阶段：骨架完成，M1 未启动

## 当前状态

- 计划文档 `docs/plans/PLAN-20260729-project-skeleton-v1.md` 已确认（运行环境 Windows PC、首发黑神话悟空、仅截屏+模拟输入、LLM 默认 Ollama 兼容 Kimi/OpenAI）
- Python 工程骨架已落地：契约层（core/games/llm 接口）、入口 stub、配置示例、契约冒烟测试

## 最近关键结论

- 共享数据契约（GameState/Action/Suggestion）放 `core/contracts.py`，保证依赖方向 apps → games → core 不倒置
- 2070s（8GB）显存约束：实时感知只许轻量模型；Ollama 仅离线复盘时运行
- 幻唐志 PK 辅助暂缓（场景不足），入口保留

## 下一步动作

- M1 启动：与用户选定黑神话首个全自动场景（候选：自动清小怪循环、自动跑图采集）
- 实现 `games/wukong` 适配器 + `core` 感知/控制最小实现

## 阻塞项

- 开发机为 Linux，目标运行环境 Windows——截屏/输入模拟（mss/pydirectinput）的实机验证需在 Windows 侧进行

## 未证实风险

- 轻量 CV 模型与游戏共存于 8GB 显存的实际推理帧率未验证
- pydirectinput 对黑神话悟空的输入兼容性未验证（部分游戏只认特定输入路径）

## 验证

- 已执行：`python -m compileall`（通过）、`pytest tests/test_contracts.py`（2 passed）
- 未执行：Windows 侧截屏/输入实机验证（环境不具备）
