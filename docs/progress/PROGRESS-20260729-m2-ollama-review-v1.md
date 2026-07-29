# 进度：M2 Ollama 周期复盘落地

- 日期：2026-07-29
- 阶段：M2 代码实现完成，待 Windows + Ollama 实机联调

## 当前状态

- M2 计划 `docs/plans/PLAN-20260729-m2-ollama-review-v1.md` 全部实现完成
- 单元测试 101 passed（Linux，Ollama 全 mock）
- M1 实机联调修复累计：窗口定位、WGC 双事件/阻塞 start、标题空格、转向平滑、探索抗摇摆、--probe-input

## 最近关键结论

- 采样策略（代理设计，用户授权）：事件帧（状态转移/接敌/死亡）+ 周期帧（300 tick≈30s，正式跑也采）+ 异常帧（脱困触发/hp 单 tick 跌幅>0.3）；帧按 tick 序号与 JSONL 前后各 30 tick 操作日志配对
- 复盘链路：`--review runs/<ts>` → 分批（4 帧/批）送 Ollama 视觉模型（默认 qwen2.5vl:7b）→ JSON 三级解析（失败降级文本）→ ReviewReport + `tuning_suggestion.yaml` 补丁
- 红线保持：LLM 不进实时决策；补丁绝不自动合入 configs，人确认后手动应用；复盘默认手动触发（2070s 显存，游戏退出后跑）

## 下一步动作

- 用户 Windows 实机：`git pull` → `.venv\Scripts\pip install -e .`（装 ollama SDK）→ `ollama pull qwen2.5vl:7b`
- 正式跑一段 → `python -m apps.auto_player --game wukong --review runs/<ts>` → 审补丁 → 手动合入 configs
- M1 实机验收报告仍缺（docs/reports/），联调顺畅后补

## 阻塞项

- 无（等实机环境）

## 未证实风险

- Ollama SDK 真实调用返回结构、qwen2.5vl:7b 在 8GB 显存的加载与速度未验证
- 视觉模型对游戏画面的诊断质量未知，prompt 可能需按实机输出迭代
- hp_drop_alert 基于血条测量，读数噪声可能误采异常帧

## 验证

- 已执行：`pytest` 101 passed（配对窗口/分批/JSON 解析降级/补丁生成/CLI 报错/正式跑三类采样帧名断言）；`compileall` 通过；`--help` 含 --review
- 未执行：Ollama 真实调用、视觉模型诊断质量、端到端复盘工作流（均需 Windows + Ollama）
