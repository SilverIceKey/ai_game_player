# 训练链路落地计划（Phase 1：Tiny Overfit）

## 背景

用户已用 `app.observe_train` 在 Windows 游戏机实机录制了第一段数据（sessions/ 结构），训练也在该游戏机上跑（有 CUDA GPU）。当前 `model/` 只有协议 + 占位 Policy，`train/trainer.py` 在无 torch 时报错——**训练代码尚不存在**。本轮目标 = spec §42 Phase 1：

> 30~60 分钟数据上，模型能够在训练数据上明显拟合玩家动作；做不到 = pipeline 有问题。

## 设计决策

1. **依赖**：新增 optional extra `train = ["torch", "torchvision"]`（不写死版本，游戏机上按 PyTorch 官方指引装 CUDA wheel：`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`）。开发机 .venv 装 CPU 版用于跑单测。torch 全部延迟导入，`load_policy`/trainer 在无 torch 时保留现有明确报错。
2. **模型（spec §16，第一阶段小模型起点）**：
   - Visual Encoder：torchvision 预训练 **ResNet18**（去 fc → 512 维/帧；遵守 §18 禁止随机初始化训视觉，默认 freeze backbone，配置项 `train_stage` 预留三阶段）
   - Action 编码：NormalizedAction → 18 维向量（4 连续轴 + 14 按钮 0/1）
   - Temporal Encoder：**GRU**（输入 = 帧 embedding 拼接当前 action 向量，history_frames=16 步）——比 Transformer 小且稳，第一阶段够用（2026-08-13 补充：应用户要求加了 `training.temporal: gru|transformer` 双实现；基准显示 16 步窗口下两者速度无差异，计算全在 backbone，默认仍 gru）
   - Policy Decoder：MLP → 未来 4 步，每步拆 Head（§19）：
     - Movement Head：2 维 tanh，MSE regression
     - Camera Head：**discretized bins**（每轴 21 bins，cross entropy；§19.2 明确 MSE 会左右互消，优先离散分布）
     - Button Head：14 个 multi-label logits，BCEWithLogits + **pos_weight 自动统计**（§24 类别不平衡）
3. **Loss（§23）**：`L = L_move + λc·L_camera + λb·L_button + λt·L_temporal`，权重取 `config.loss_weights`；L_temporal = chunk 内相邻步动作差的平滑惩罚。全部写进 checkpoint meta.training_config（§29 可复现）。
4. **数据管线**：复用 `dataset/sample_builder.py` 时间对齐（含 §12 offset）与 `runtime/preprocess.py`（与推理同一预处理，384×216）；新增 torch Dataset 包装：sessions_dir 下所有 session → EpisodeStoreReader → build_samples → 懒加载帧（Cv2FrameLoader）→ tensor。DataLoader shuffle 训练；§28 Replay Buffer 类别加权留到 DAgger 阶段（当前只有 human 数据，权重无意义）。
5. **帧归一化一致性**：训练与推理共用 `model/encoding.py` 的张量化/反张量化函数（ImageNet mean/std normalize、camera bin 编解码），避免 train/inference 漂移。
6. **checkpoint**：`checkpoints/<model_version>/model.pt`（state_dict）+ `meta.json`（ModelCheckpointMeta：model_version/dataset_version/code_commit/training_config/eval_result，§29）；训练后自动 `ModelRegistry.register_candidate`（§7）。
7. **load_policy 落地**：有 checkpoint + torch 时返回 `TorchPolicy`（实现 VideoActionPolicy 协议：np 帧 + action history → ActionChunk；camera 取 bin 期望，按钮 sigmoid>0.5），AUTOPILOT/SHADOW 即刻可用真实模型。
8. **CLI**：`python -m app.train --sessions sessions/ --epochs N --batch-size B --lr ...`（默认值进 config.py 新 `training` 段）；跑完打印 train loss 曲线 + 训练集上的 §36 指标（movement/camera error、按钮 P/R），作为 Phase 1 过拟合判据。

## 明确不做（本轮）

- 三阶段 unfreeze 实际调度（配置字段预留，默认 freeze）
- 固定冻结 eval set 的离线评估（数据量不够；先用训练集指标验证过拟合）
- DAgger 迭代与 correction 加权采样、closed-loop 评估
- 模型规模搜索（§17 区间是后续实验起点，不是本轮目标）

## 实施步骤

1. 计划文档落盘 `docs/plans/PLAN-20260813-training-pipeline-v1.md`
2. `pyproject.toml` 加 `train` extra；开发机 .venv 装 CPU 版 torch/torchvision（仅测试用）
3. `config.py` 加 `TrainingConfig`（epochs/batch_size/lr/camera_bins/train_stage）+ settings.example.yaml 补 `training:` 段
4. `model/encoding.py`：NormalizedAction ↔ tensor、camera bins 编解码（训练/推理共享）
5. `model/torch_model.py`：VideoActionNet（ResNet18 + GRU + 三 Head）
6. `train/dataset.py`：torch Dataset（sessions → samples → tensors，帧懒加载）
7. `train/losses.py`：§23 组合 loss + §24 按钮 pos_weight 统计
8. `train/trainer.py` 落地 `train_candidate`（构造参数改为接受 torch Dataset 工厂；更新 test_trainer.py）
9. `model/policy.py` `load_policy` 落地 + `model/torch_policy.py`（TorchPolicy）
10. `app/train.py` CLI（训练 → checkpoints/<version>/ → registry.register_candidate → 打印 §36 训练集指标）
11. 测试：前向形状、loss 数值、camera bin 往返、合成数据单步过拟合（loss 下降）、checkpoint save/load 往返、TorchPolicy 输出契约、CLI 装配（假数据小跑）
12. 验证：`.venv/bin/python -m pytest -q` 全绿 + compileall
13. 文档收口：progress 新增、agent-context 更新、README 加训练入口说明（含游戏机 CUDA 安装命令）

## 验证与风险

- 开发机只能验证 CPU 小模型单测；**真实数据上的 Phase 1 过拟合需游戏机上执行**（给出确切命令与判据：train loss 显著下降 + 训练集按钮 P/R 明显高于随机）
- 风险：mp4v 有损帧对训练的影响未知；43k 样本/小时 @12fps 的 DataLoader 解码速度未实测（慢则先做帧缓存）；游戏机显存未知（ResNet18+384×216 batch 32 约 <2GB，一般够用）
