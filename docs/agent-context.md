# 交接上下文

## 当前状态

- 当前主任务：M2——Ollama 周期复盘与策略调参（代码完成，待实机联调）
- 当前阶段：Windows + Ollama 实机联调（M1 校准验收并行进行中）
- 当前结论：
  - M2 全部代码落地，`pytest` 101 passed（Ollama 全 mock）
  - 采样策略：事件帧 + 周期帧（300 tick≈30s，正式跑也采）+ 异常帧（脱困/hp 骤降），帧与前后 30 tick 操作日志配对
  - `--review runs/<ts>` 产出复盘摘要 + tuning_suggestion.yaml 补丁（不自动合入）

## 本轮改动

- 新增 `docs/plans/PLAN-20260729-m2-ollama-review-v1.md`（已确认）
- 新增 `llm/providers/ollama_provider.py`、`llm/review/engine.py`、`llm/review/prompts.py`、`llm/tuning/patch.py`
- 修改 `apps/auto_player/main.py`（正式跑采样落帧 + `--review` 模式）、`core/recorder/jsonl.py`（replay.jsonl + 读取配对）、`core/config.py`（vision_model + review 段）、`core/decision/navigation.py`（unstick_triggered 信号）、`configs/settings.example.yaml`
- 新增 `tests/test_review.py`（17 例）；适配 replay.jsonl 改名
- 感知设计修正（M1 计划 3.1a/3.1b）：新增 `core/perception/bars.py`（动态血条检测）、`apps/auto_player/edit_roi.py`（--edit-roi 交互校准）；adapter 改双色动态敌条 + hp_visible；calibrate 适配；pyproject 平台分流 opencv + ruamel.yaml；测试 101 → 115 例
- 更新 `docs/progress/PROGRESS-20260729-m2-ollama-review-v1.md`

## 验证结果

- 已执行：`pytest` 101 passed、`compileall` 通过、`--help` 含 --review
- 未执行：Ollama 真实调用、视觉模型诊断质量、端到端复盘（需 Windows + Ollama）
- 证据：`.venv/bin/python -m pytest -q` → 101 passed in 0.91s

## 风险与限制

- Ollama SDK 返回结构、qwen2.5vl:7b 在 2070s 8GB 的加载未验证（游戏退出后跑）
- 视觉模型诊断质量未知，prompt 或需迭代；hp_drop_alert 可能误采
- M1 实机风险同前（HUD 校准、光流漂移、CPU 占用）

## 下一步

- 用户实机：pull → `pip install -e .` → `ollama pull qwen2.5vl:7b` → 跑一段 → `--review runs/<ts>` 验收
- M1 实机验收报告补落 docs/reports/
