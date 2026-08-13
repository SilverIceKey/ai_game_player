# PROGRESS-20260813-training-pipeline-v1：训练链路落地（Phase 1）

> 计划：`docs/plans/PLAN-20260813-training-pipeline-v1.md`；规格：`docs/AI_Game_Player_SPEC_v1.0.md`

## 当前状态

- 当前主任务：训练链路落地（spec §42 Phase 1 tiny overfit）——**代码完成，待游戏机真实数据验证**
- 触发背景：用户已用 `app.observe_train` 实机录制第一段数据；训练在 Windows 游戏机（CUDA）本地跑
- 当前结论：`python -m app.train` 全链路可用：sessions/ → 样本构造（§12/§22）→ ResNet18(frozen)+GRU+三 Head（§16/§19）→ §23 组合 loss + §24 pos_weight → checkpoints/<version>/ + registry candidate（§7/§29）

## 本轮改动

- 新增 `model/encoding.py`（动作向量/camera bin/帧归一化，训练推理共享）、`model/torch_model.py`（VideoActionNet，§18 三阶段冻结）、`model/torch_policy.py`（TorchPolicy 推理 + checkpoint 加载）
- 新增 `train/dataset.py`（SessionDataset 懒加载帧）、`train/losses.py`（§23/§24）、`app/train.py` CLI
- 落地 `train/trainer.py`（替换骨架：真实训练循环 + 训练集 §36 指标 + checkpoint 落盘）、`model/policy.py` `load_policy`（有 checkpoint 时加载真实模型）
- `config.py` 加 `TrainingConfig`（epochs/batch_size/lr/camera_bins/train_stage）；settings.example.yaml 补 `training:` 段
- `pyproject.toml` 加 `train` extra（torch/torchvision）；开发机 .venv 装 CPU 版用于单测
- 测试 +30：encoding/torch_model/losses/train_dataset/trainer/torch_policy/train_cli（合成 session 端到端小训练，CPU）
- 训练日志加详细进度（2026-08-13 补充）：pos_weight 预扫描提示、batch 级进度行（≥5% 或 3s 心跳，含 running loss / samples/s / ETA）、epoch 耗时行、评估阶段提示——2070s 级别显卡上不再"看着像卡死"

## 验证结果

- 已执行：`pytest` **297 passed**；compileall；`python -m app.train --help`；合成数据端到端训练（loss 下降、checkpoint 落盘、registry candidate、TorchPolicy 加载推理契约）
- 未执行：游戏机真实数据训练（需 CUDA torch）；真实 checkpoint 驱动 AUTOPILOT
- 证据：`.venv/bin/python -m pytest tests/ -q` → 297 passed in 9.47s

## 风险与限制

- mp4v 有损帧对训练精度的影响未知；DataLoader 视频懒解码速度未实测（慢则做帧缓存）
- num_workers=0 单进程加载（Windows 多进程 pickle 视频句柄不稳），吞吐瓶颈待实测
- Action History 参与方式是 mean-pool 18 维向量（最简方案），表达连段上下文的能力有限，Phase 2 不够再升级
- 训练与推理的预处理一致性靠共用 model/encoding.py 保证，实机需抽查

## 下一步

1. 游戏机：`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`
2. `python -m app.train` 跑 Phase 1，核对判据（loss 下降 + 按钮 P/R 超随机）；不过则按 §17 排查（先查时间同步）
3. 过了 Phase 1 → `autopilot --checkpoint ... --dry-run` 看动作 → Phase 3 SHADOW 实机对齐
