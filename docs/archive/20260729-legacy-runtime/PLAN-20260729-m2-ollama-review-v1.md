> **已归档**（2026-08-12）：本文档属于已弃用的「ROI 感知 + FSM 决策 + LLM 复盘」旧路线，不再适用。
> 替代文档：`docs/AI_Game_Player_SPEC_v1.0.md` 与 `docs/plans/PLAN-20260812-spec-v1-refactor-v1.md`。

# M2 计划：Ollama 周期复盘与策略调参

- 日期：2026-07-29
- 状态：**已确认**（用户提出需求并授权采样策略由代理设计）
- 上游文档：`docs/plans/PLAN-20260729-project-skeleton-v1.md`（LLM 只做离线复盘调参，不进实时决策）

## 1. 范围

本轮做：

- 运行中按策略**采样帧 + 配对操作日志**（非全量画面）
- `--review <run_dir>`：Ollama 视觉模型对采样帧 + 操作日志做复盘分析
- 输出复盘报告 + `configs` 参数调整建议（YAML 补丁文件，**人确认后手动应用**）

本轮不做：

- 游戏运行中实时调用 Ollama 改策略（2070s 显存约束 + LLM 不进实时链路的架构红线）
- Kimi / OpenAI provider（接口已留，后续接）
- 自动应用参数变更（必须人确认）

## 2. 采样策略（代理设计，用户授权）

原则：**事件驱动为主，周期保底，异常加采**——不采全量画面。

| 类型 | 触发条件 | 默认 |
| --- | --- | --- |
| 事件帧 | FSM 状态转移、首次接敌、死亡 | 已有机制 |
| 周期帧 | 每 `sample_interval_ticks` 落一帧 | 300 tick ≈ 30 秒 @10fps |
| 异常帧 | 卡住脱困触发、hp 单 tick 跌幅 > `hp_drop_alert` | 0.3 |

- 正式跑（非 dry-run）也开启采样；dry-run 沿用已有 `frame_interval_ticks`
- 帧文件名含 tick 序号（已有约定 `{tick:06d}_{tag}.png`），复盘时按序号从 JSONL 回放中取**前后各 K=30 tick** 的 state/intent/action 与该帧配对——"画面 + 操作"作为一个分析单元

## 3. 复盘链路

```text
runs/<ts>/（frames/*.png + replay.jsonl + session.log）
  → llm/review：按采样策略配对帧与操作窗口，分批送 Ollama（每批 4 帧）
  → 每批输出：判断是否合理、异常诊断、参数建议（JSON）
  → 汇总为 ReviewReport（llm/base.py 已有契约）
  → llm/tuning：tuning_suggestions 写成 runs/<ts>/tuning_suggestion.yaml（补丁）
  → 人审阅后手动合入 configs/wukong.yaml
```

- Provider：`llm/providers/ollama_provider.py`，ollama SDK（`chat(images=[...])`）
- 模型：`configs/settings.yaml` 的 llm 段新增 `vision_model: "qwen2.5vl:7b"`（图文一起送）；8GB 显存在游戏退出后跑 7B Q4 可行
- prompt 模板放 `llm/review/prompts.py`，要求模型输出固定 JSON 结构，解析失败降级为纯文本摘要
- 分批是为控制上下文与显存峰值；批大小走配置 `review.batch_size: 4`

## 4. 实现清单

| 文件 | 内容 |
| --- | --- |
| `apps/auto_player/main.py` | 正式跑周期/异常落帧；新增 `--review <run_dir>` 模式 |
| `core/recorder/jsonl.py` | 回放读取支持（按 tick 窗口取记录），供配对 |
| `llm/providers/ollama_provider.py` | Ollama Provider（文本+图片），延迟导入 ollama |
| `llm/review/engine.py` | 采样配对、分批调用、汇总 ReviewReport |
| `llm/review/prompts.py` | prompt 模板（帧+操作窗口+当前参数 → JSON 诊断） |
| `llm/tuning/patch.py` | tuning_suggestions → YAML 补丁文件（不自动应用） |
| `configs/settings.example.yaml` | llm 段加 vision_model、review 段（batch_size、window_ticks、hp_drop_alert、sample_interval_ticks） |
| `tests/` | provider mock、配对逻辑、JSON 解析降级、补丁生成 |

## 5. 约束与风险

- ollama 延迟导入，Linux 开发机测试全 mock；Ollama 实机联调在 Windows
- LLM 建议只写补丁文件，不自动改 configs（人确认红线）
- 2070s：复盘必须在游戏退出或低负载时跑；`--review` 默认手动触发，不做会话结束自动拉起（避免抢显存）
- 视觉模型对游戏画面的诊断质量未验证，第一版 prompt 可能需要按实机输出迭代

## 6. 验收

- 单元测试全绿（mock provider 下的配对/解析/补丁）
- 实机：`--dry-run` 或正式跑一段 → `python -m apps.auto_player --game wukong --review runs/<ts>` → 产出复盘摘要 + tuning_suggestion.yaml
