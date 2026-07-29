# 交接上下文

## 当前状态

- 当前主任务：M1——黑神话悟空全自动最小链路（未启动）
- 当前阶段：骨架已完成并验证，待选定 M1 首个场景
- 当前结论：
  - 计划文档 `docs/plans/PLAN-20260729-project-skeleton-v1.md` 已确认
  - 工程骨架：契约层（`core/contracts.py`、`core/*/base.py`、`games/base.py`、`llm/base.py`）+ 入口 stub + 配置示例
  - 验证：compileall 通过、`pytest` 2 passed

## 本轮改动

- 计划文档 v1 定稿（Windows PC、首发黑神话、仅截屏+模拟输入、Ollama 默认兼容 Kimi/OpenAI、2070s 显存约束）
- 新增 Python 骨架：`core/`（contracts + perception/decision/control/recorder 接口）、`games/`（base + wukong/genshin/ananta 插件位）、`llm/`（base + providers/review/tuning 位）、`apps/`（auto_player / pk_assistant 入口）
- 新增 `pyproject.toml`、`configs/settings.example.yaml`、`.env.example`、`tests/test_contracts.py`
- `.gitignore` 增加 `runs/`（回放输出）

## 验证结果

- 已执行：`python -m compileall`（COMPILE_OK）、`pytest -q`（2 passed）
- 未执行：Windows 实机截屏/输入验证（开发机为 Linux）
- 证据：`.venv/bin/python -m pytest -q` 输出 2 passed

## 风险与限制

- mss/pydirectinput 实机行为只能在 Windows 验证
- 轻量 CV 模型与游戏共享 8GB 显存的帧率未验证
- pydirectinput 对黑神话的输入兼容性未验证

## 下一步

- 与用户选定 M1 首个场景（候选：自动清小怪循环、自动跑图采集）
- 提交本轮骨架改动到 git（待用户确认）
