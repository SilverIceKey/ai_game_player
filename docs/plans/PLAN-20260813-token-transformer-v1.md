# 计划：移除 GRU/LSTM，统一 Token Transformer Temporal Policy（含 Memory）

> 依据用户 40 条指令。先影响分析（指令 #39），后改动清单。**不做向前兼容，旧 GRU/LSTM checkpoint 明确报 legacy/unsupported，不 silent fallback。**

## 一、影响分析（指令 #39）

### 现状

- **三种时序实现所在文件**：`model/torch_model.py`（`TEMPORAL_ENCODERS` + `temporal` 构造参数 + forward 分支）；配置 `config.py` `TrainingConfig.temporal` + `_load_training` 校验；快照 `train/trainer.py:107,264`；加载 `model/torch_policy.py:133`；测试 `tests/test_torch_model.py`（gru/lstm/transformer/invalid 四例）；示例 `configs/settings.example.yaml:46`。**runtime 无分支**（InferenceWorker 只调 policy.predict）。
- **Temporal 调用链**：`app.train → Trainer → VideoActionNet(temporal=...)`；推理 `app.autopilot → load_policy → load_torch_policy（按 meta.training_config 重建）→ TorchPolicy.predict ← InferenceWorker`；SHADOW 同走 InferenceWorker。
- **Visual Encoder 输出 shape**：ResNet18 去 fc（含 avgpool）→ `(B*k, 512, 1, 1)` → `(B,k,512)`。**全局池化已丢失空间信息**，Token Compressor 需要改为去两层（fc+avgpool）→ `(B*k, 512, 6, 12)` @384×216。
- **当前 Transformer 输入 shape**：`(B, 16, 512)` 帧 embedding 序列 + 共享 learned pos_embed；action history 只是 mean-pool 成 18 维 concat 到 decoder（不是 token）。
- **Action History 当前表示**：最近 m=16 条 ActionRecord（事件驱动）→ 每条 `action_to_vector` 18 维 → `(B,m,18)` 左 pad → mean-pool。无年龄信息进模型。
- **Action Decoder 当前 shape**：decoder MLP `(B, 530)→256→256`；三头：move `(B,4,2)` tanh / camera `(B,4,2,21)` logits / button `(B,4,14)` logits。
- **Checkpoint 兼容风险**：旧 checkpoint `training_config.temporal="gru"|"lstm"|"transformer"(旧2层)`，结构与新 Token Transformer 完全不同 → **不可兼容**，loader 对无 `"arch": "token_transformer_v1"` 标记的 checkpoint 明确报 legacy/unsupported。settings.yaml 残留 `training.temporal` 键 → ConfigError 提示删除（migration 提示，不实例化）。
- **需要修改的文件**：`model/torch_model.py`（重写主体）、`model/torch_policy.py`、`model/policy.py`（协议签名）、`model/visual_encoder.py`/`model/temporal_policy.py`（接口文档注释更新）、`config.py`、`configs/settings.example.yaml`、`train/trainer.py`、`train/dataset.py`（action 增强 + memory 帧）、`train/losses.py`（不动 loss 本体，trainer 记录 gate 统计）、`runtime/ring_buffer.py`（ActionHistoryBuffer 返回带时间戳记录）、`runtime/inference.py`（传时间戳 + receding horizon 节奏）、`app/autopilot.py`（execute_steps 截断提交）、`evaluation/dependency.py`（新）、`evaluation/offline.py`（按场景分组）、测试若干。

## 二、新架构设计（对齐指令 #2/#4-#16/#21-#28/#33-#35）

### Token Schema（#4/#26/#27）

统一 token 序列（单一样本）：

```text
[VISUAL × (k·Kt)] + [ACTION × m] + [MEMORY × S]     L = k·Kt + m + S
```

每个 token = `ContentEmbedding + TypeEmbedding + SpatialEmbedding(仅 VISUAL)`，另加 **key 侧 age attention bias**（等价 relative temporal encoding，指令 #27 允许等价方案）：

```text
attn_mask[b, i, j] = -λ_type(j) · age_j(秒)          # (B·heads, L, L)，nn.MultiheadAttention 支持
λ: action(快) > visual(中) > memory(慢)，全部配置化   # #8/#9/#12：decay 是 prior，attention 仍可学
```

- `token_type`：learned type embedding（3 类），不靠位置猜（#26）
- `temporal_position`：age bias 承载相对时间；VISUAL 另加每帧内 spatial slot learned embedding
- `token_age`：训练侧 = anchor_us − 记录时间戳；推理侧同理（统一 now_us 时钟）
- `source`：由 type 承载

### Visual Token（#5/#28）

```text
Frame (384×216) → ResNet18 去 fc/avgpool → (512, 6, 12) spatial
→ AdaptiveAvgPool 到 grid（Kt=4→2×2, 8→2×4, 16→4×4）→ Kt tokens × d_model
```

`transformer.visual_tokens_per_frame` ∈ {4, 8, 16}（构造校验，非法值报错）。16 帧 × 8 = 128 visual tokens，不爆炸（#28）。

### Action Token（#7/#18）

- 最近 m 条 ActionRecord（默认 history_actions=8，示例 yaml 改 8，注释 200~500ms 定位）→ 每条 18 维 → Linear→d_model token，年龄进 bias。
- 训练增强（SessionDataset，仅训练）：`action_history: {dropout_prob: 0.25, mask_prob: 0.20, random_truncate: true}`——dropout=随机整步清零；truncate=随机砍前缀。配置化（#18）。

### Memory（#10-#13/#34）

- **压缩路径（训练/推理同一规则）**：一帧 → backbone+compressor → Kt tokens → per-frame pool+MLP（MemoryWriter）→ 1 个 slot (d_model)。**不保存 raw frame/action 进 Memory**，slot 由 temporal representation 压缩得到。
- **训练侧**：样本取最近窗口之前、按 `memory.update_interval_ms` 网格回溯 S 帧（S=slots=16，500ms → ≈8s；拉长时间尺度调 update_interval_ms 即可，如 2000ms→32s），逐帧压缩成 S slots，年龄=anchor−帧时间戳；历史不足 → 零 slot + 大年龄（bias 近 −∞ 等效屏蔽）。
- **推理侧（TorchPolicy 持有 runtime state）**：每 `update_interval_ms` 用最新帧跑一次 backbone+compressor+writer → push (slot_vec, ts) 进 deque(maxlen=S)；predict 时组 (S,d) 张量。**Memory 是 runtime state 不是 weight**（#34）：checkpoint 只存 writer 参数；`reset_memory()` 显式 API（死亡/读档/主菜单/手动 → 调用方清 memory + Action History + scheduler chunk，#13 保留 API，死亡检测不在本轮）。
- **代价明示**：训练每样本多 S 次 backbone 前向（backbone 计算 ≈ ×2），已测试可接受；slots/update_interval 可调配。

### Temporal Transformer（#25）

`nn.TransformerEncoder`：d_model=hidden_dim(512)、num_layers=6、num_heads=8、dropout=0.1，全部配置化（示例 yaml 即用户给的初始值）。输入 L×d_model + 上述 mask。

### Learned Gate（#14-#17）

```text
z_cur  = mean(全部输出 token)
z_fast = mean(VISUAL+ACTION 输出)
z_slow = mean(MEMORY 输出)
[g_f, g_s] = sigmoid(Linear(z_cur))      # 连续值，不强制归一，无人类语义
z = z_cur + g_f·z_fast + g_s·z_slow → 既有三头 decoder → Action Chunk [B,H,·]
```

- 训练每 epoch 记录 fast/memory gate 的 mean/std；`mean>0.99 & 另一<0.01`（或反向）→ eval_result 标 `possible_gate_collapse: true`（#17）
- 无 memory 分支时 gate 退化为单路（memory.enabled=false 时 S=0，z_slow 不参与，结构由快照驱动）

### Future Latent Head（#24）

本轮**只预留接口**：配置 `transformer.future_latent_head: false`，置 true 时构造即 `NotImplementedError`（没有合理 latent target，不硬造错误实现）。AUTOPILOT 无此开销。

### Receding Horizon（#22）

- `prediction.execute_steps: 2`（≤future_action_steps，校验）；InferenceWorker 推理节奏 = `step_ms × execute_steps`；AUTOPILOT 提交 scheduler 前把 chunk 截断为前 execute_steps 步。

### Runtime 接口（#33/#35）

- `TorchPolicy.predict(frames_with_ts, action_records, audio_pcm)`：frames 改传 `(frame, timestamp_us)` 对、action 改传 ActionRecord（年龄需要时间戳）→ `model/policy.py` 协议签名更新，Placeholder/Random 签名跟随（忽略内容）。InferenceWorker/ring_buffer 相应改（`ActionHistoryBuffer.recent_records`）。
- diagnostics：`policy.last_diagnostics` = {fast_gate, memory_gate, token_counts, memory_slots_filled}；进 inference.jsonl（attention summary 仅 debug：不取 attention weights，gate 即摘要，明示为取舍）。延迟 p50/p95/p99 已有（LatencyStats）；VRAM 在 cuda 时记 `torch.cuda.mem_get_info`；memory update/reset 计数进 diagnostics。

### 依赖测试（#19/#20）

新 `evaluation/dependency.py`：`dependency_report(net, loader, device)` 在训练结束后跑（样本子集），输出各消融的 move/camera 误差与按钮 P/R 对比：

```text
normal / shuffled_video / zero_action / shuffled_action / zero_memory / shuffled_memory
```

写进 eval_result["dependency"]；解读规则（shuffle video 无影响=没用视觉等）打印提示但不自动下结论。

### 场景拆分评估（#30-#32，范围说明）

- 数据无场景标签，**不强造**：episode meta 支持可选 `scene` 字段（人工在 episodes.json 标注或目录约定），`evaluation/offline.py` 加 `evaluate_samples_by_scene()`——有标签分组输出 action 指标 + gate 分布 + memory 消融 delta；无标签退化为全局。
- Dodge/Heal Recall = 对应按钮的 per-button P/R（已有）。Damage Taken / Survival / Progress Distance 等需要游戏状态读取或闭环实跑，本轮**不实现**，文档明示 NOT RUN。

## 三、移除清单（#1/#37）

- `model/torch_model.py`：删 `TEMPORAL_ENCODERS`/`temporal` 参数/GRU/LSTM 分支，整体重写为 Token Transformer（类名保留 `VideoActionNet`，ctor 换参数集）
- `config.py`：删 `TrainingConfig.temporal`；`_load_training` 遇到 `temporal` 键 → ConfigError（legacy migration 提示）；新增 `TransformerConfig`/`MemoryConfig`/`ActionHistoryConfig`；`PredictionConfig.execute_steps`
- `train/trainer.py`：删 temporal 传参，快照改存新结构参数 + `"arch": "token_transformer_v1"` + gate 统计
- `model/torch_policy.py`：删 temporal 重建分支；无 arch 标记 → RuntimeError("legacy/unsupported")
- `configs/settings.example.yaml`：删 `training.temporal`，新增 `transformer:`/`memory:`/`action_history:` 段
- 测试：删 gru/lstm/invalid-temporal 三例，改写新架构测试
- 文档：spec §16 重写为 Token Transformer 架构、§8.2/§8.3 按新职责改写；README 模型章节同步

检查项：grep 确认无 `gru`/`lstm` 残留分支（代码与测试），docs 同步。

## 四、配置 Schema（新增，全部配置化）

```yaml
prediction:
  execute_steps: 2          # receding horizon：预测 4 执行 2
transformer:
  hidden_dim: 512
  num_layers: 6
  num_heads: 8
  dropout: 0.1
  visual_tokens_per_frame: 8   # 4/8/16
  future_latent_head: false    # 预留接口，true 即报错
  age_decay: {action: 2.0, visual: 0.5, memory: 0.05}   # λ/秒
memory:
  enabled: true
  slots: 16
  update_interval_ms: 500      # 同时决定训练侧 memory 帧网格（2Hz）
action_history:
  dropout_prob: 0.25
  mask_prob: 0.20
  random_truncate: true
```

`model.history_actions` 沿用为 action token 数（示例改 8）；`model.sample_fps/history_frames` 不动（即指令 #6 的 video_history，已有配置不重复造名）。

## 五、明确不做（本轮）

- World Model / Planning / Diffusion 等（#23 用户排除）
- Future Latent Head 真实实现（#24 允许只预留）
- 死亡/读档等 Memory Reset 自动检测（#13 保留显式 API）
- 场景标签自动打标；combat 闭环指标（Damage Taken 等需游戏状态，无读取手段）
- attention 权重矩阵导出（#33 允许省略，用 gate 值当 summary）
- 3D spatial RoPE（visual token 是压缩后的少量 token，age bias 即相对时间编码）

## 六、实施步骤

1. 本计划落盘 `docs/plans/PLAN-20260813-token-transformer-v1.md`；spec §16/§8.2/§8.3 更新
2. `config.py` + settings.example.yaml（新段 + legacy 报错）
3. `model/torch_model.py` 重写（token schema/visual compressor/memory writer/gate/heads）
4. `train/dataset.py`（memory 帧进缓存、action 增强、年龄张量）+ `train/trainer.py`（gate 统计、快照）
5. `model/torch_policy.py`（runtime memory state、diagnostics、legacy 拒绝）+ `model/policy.py` 协议 + `runtime/ring_buffer.py`/`runtime/inference.py` + `app/autopilot.py`（execute_steps）
6. `evaluation/dependency.py` + `offline.py` 场景分组 + app.train 收尾调用
7. 测试：新 token schema/gate/age bias/memory 缓冲/receding horizon/dependency/legacy 拒绝/config；全量 pytest + compileall
8. 文档收口：progress、agent-context、README；`grep -ri "gru\|lstm" --include=*.py .` 确认零残留

## 七、验证与风险

- 开发机：全量单测（CPU，含新架构前向形状、age bias 数值、gate 连续性与统计、memory 槽位 FIFO/reset、legacy checkpoint 报错、消融报告结构）+ 合成数据端到端小训练
- **不可验证项如实标注**：Inference P50/P95/P99、VRAM（无 GPU，"未验证"）；真实战斗/探索场景 gate 行为（需游戏机数据 + 实跑）
- 风险：memory 训练/推理路径虽同规则但训练时 memory 内容来自同视频旧帧（非模型在线生成），存在分布差（dependency test 的 zero/shuffled_memory 用于暴露）；backbone 计算量约 ×2（16+16 帧/样本），2070s 上 epoch 时间翻倍左右，可用 slots/input 分辨率调；gate collapse 有统计兜底但不保证学好
