# PROGRESS-20260813-token-transformer-v1

## 当前状态

- 当前主任务：GRU/LSTM 移除，统一 Token Transformer Temporal Policy（spec §16 v1.0 修订）——代码完成，待游戏机真实训练验证
- 当前阶段：架构改造收口；旧 GRU/LSTM/早期 transformer checkpoint 一律 legacy/unsupported（loader 按 `training_config.arch` 拒绝，不 silent fallback），需重新训练
- 当前结论：
  - 唯一时序实现 = Token Transformer：`model/torch_model.py`（ARCH_TAG=`token_transformer_v1`）
  - Token 流：[VISUAL(k×Kt) | ACTION(m) | MEMORY(S)] + TypeEmbedding + key 侧年龄 attention bias（action 快衰减 2.0 / visual 0.5 / memory 0.05，配置化，decay 是 prior 非硬规则）
  - Memory 是 runtime 状态不是权重：训练侧样本窗口之前按 update_interval_ms 回溯 S 帧逐帧压缩；推理侧 TorchPolicy 内部 deque 每 500ms 用最新帧压 1 slot，`reset_memory()` 为 Hard Reset 显式 API（死亡/读档由 app 层判定，未自动接线）
  - Learned Gate：z = z_cur + g_fast·z_fast + g_slow·z_slow，连续 sigmoid 值、无人工语义；训练每 epoch 记录 mean/std，单极化标 `possible_gate_collapse`
  - Receding Horizon：`prediction.execute_steps=2`，预测 4 步执行 2 步；推理节奏与 chunk 截断都已按此改
  - 依赖消融：训练收尾自动跑 `evaluation/dependency.py`（normal/shuffled_video/zero_action/shuffled_action/zero_memory/shuffled_memory），结果写 eval_result["dependency"]

## 本轮改动

- `config.py`：删 `TrainingConfig.temporal`（遇到旧键报 ConfigError）；新增 TransformerConfig / MemoryConfig / ActionHistoryConfig；PredictionConfig 加 execute_steps
- `model/torch_model.py`：整体重写为 Token Transformer（TokenCompressor learned query 交叉注意力、MemoryWriter、TypeEmbedding、age bias mask、gate、预留 future_latent_head 接口（True 即 NotImplementedError））
- `model/torch_policy.py`：predict 改收 (frame,ts) 对 + ActionRecord 列表；runtime memory deque；last_diagnostics；loader arch 校验
- `model/policy.py`：协议签名更新（Placeholder/Random 跟随）
- `train/dataset.py`：样本加 frame_ages/action_ages/memory_slots；Action History 增强（dropout/mask/truncate，`dataset.augment` 开关，评估时关闭）
- `train/trainer.py`：新 forward 参数；gate 统计 + collapse 告警；收尾接 dependency_report；快照加 arch/transformer/memory/execute_steps
- `runtime/inference.py` / `runtime/ring_buffer.py` / `app/autopilot.py`：带时间戳传递、recent_records()、execute_steps 节奏与 chunk 截断、diagnostics 入推理日志
- `evaluation/dependency.py`（新）/ `evaluation/offline.py`（加 evaluate_samples_by_scene，场景标签可选、仅评估用途）

## 验证结果

- 已执行：`pytest` 320 passed；compileall 全量；gru/lstm 残留 grep（仅剩 legacy 拒绝文案）；默认配置参数量实测 31.9M
- 未执行：真实数据训练、GPU 推理延迟（开发机无 GPU）；P50/P95/P99/VRAM 未验证，禁止引用任何估算数字
- 证据：`.venv/bin/python -m pytest -q` → 320 passed

## 风险与限制

- 训练/推理 memory 分布差：训练用样本窗口前 S 帧逐帧压缩（非在线 recurrence），推理用在线 deque；靠 zero_memory 消融暴露
- gate collapse 只告警不自动纠正；Action Shortcut/Visual Ignoring 靠消融 delta 人工判读
- Token 数：16 帧×8 + 8 action + 16 memory = 152 token，全连接 attention 可控；不要再加大 history 直通 token（长程走 memory）
- 场景拆分指标（§30 评估）需要数据带 scene 标签，当前数据集没有——`evaluate_samples_by_scene` 已备，标签缺失时退化为全局

## 下一步

1. 游戏机重新训练（旧 checkpoint 全部作废）：`python -m app.train`，观察 gate 分布与 dependency delta
2. AUTOPILOT 实机验证 + P50/P95/P99/VRAM 实测
3. 若 zero_memory Δ≈0：排查 memory 训练信号（考虑 future_latent_head 辅助损失，接口已预留）
