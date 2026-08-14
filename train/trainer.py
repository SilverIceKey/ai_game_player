"""Trainer：候选模型训练（spec §23 Behavior Cloning + §7 Candidate Model）。

spec §6 禁令：采集时不允许实时 backward()（与游戏抢 GPU、破坏采集时序）。
本 Trainer 只被 app.train（离线命令）或 TrainScheduler（episode 结束后）调用，
绝不嵌入采集/推理主链路。

产物（spec §29 可复现性）：
- checkpoints/<model_version>/epochs/epoch-NNN/{model.pt,meta.json}（逐轮不可覆盖）
- checkpoints/<model_version>/final/{model.pt,meta.json} + 根 meta.json（最终选择与汇总）
- 训练集上的 §36 指标写入 eval_result，作为 Phase 1（§42 tiny overfit）判据

本模块顶层 import torch：只在训练路径上延迟导入（app.train）。
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from capture.action import BUTTONS, NormalizedAction
from capture.clock import now_us
from config import (
    AudioConfig,
    LossWeights,
    MemoryConfig,
    ModelConfig,
    PredictionConfig,
    TrainingConfig,
    TransformerConfig,
)
from model.checkpoint import ModelCheckpointMeta
from model.encoding import bin_probs_to_camera
from train.losses import compute_button_pos_weight, compute_loss

_TORCH_MISSING_MSG = (
    "模型训练需要 PyTorch：pip install torch torchvision "
    "（游戏机 CUDA 安装见 README 训练章节）"
)


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    """候选模型训练器。

    - dataset 由调用方构建（train/dataset.py SessionDataset），训练超参与
      loss 权重来自 config（§23/§29）；
    - 状态流转（candidate → promote/reject）交给 train/registry.py（§7）。
    """

    def __init__(
        self,
        loss_weights: LossWeights,
        training: TrainingConfig,
        model_config: ModelConfig,
        prediction: PredictionConfig,
        transformer: TransformerConfig,
        memory: MemoryConfig,
        device: torch.device | None = None,
        pretrained: bool = True,
        audio: AudioConfig | None = None,
    ):
        if importlib.util.find_spec("torch") is None:
            raise RuntimeError(_TORCH_MISSING_MSG)
        self._loss_weights = loss_weights
        self._training = training
        self._model_config = model_config
        self._prediction = prediction
        self._transformer = transformer
        self._memory = memory
        self._device = device or _default_device()
        self._pretrained = pretrained
        self._audio = audio

    def _audio_mels(self) -> int | None:
        """音频分支的 mel 数（spec §8.5）；audio 未开启时为 None（无音频分支）。"""
        return self._audio.mels if (self._audio is not None and self._audio.enabled) else None

    def _audio_snapshot(self) -> dict[str, int] | None:
        if self._audio is None or not self._audio.enabled:
            return None
        return {
            "sample_rate": self._audio.sample_rate,
            "mels": self._audio.mels,
            "fft_size": self._audio.fft_size,
            "hop_size": self._audio.hop_size,
        }

    def train_candidate(
        self,
        dataset: Any,
        dataset_version: str,
        model_version: str,
        code_commit: str = "",
        checkpoints_dir: str | Path = "checkpoints/",
    ) -> ModelCheckpointMeta:
        """训练候选模型并落盘 checkpoint（spec §7：禁止直接覆盖 Active Model）。"""
        if not dataset_version.strip():
            raise ValueError("dataset_version 不能为空（spec §29）")
        if not model_version.strip():
            raise ValueError("model_version 不能为空")

        from model.torch_model import VideoActionNet

        tcfg = self._transformer
        net = VideoActionNet(
            history_frames=self._model_config.history_frames,
            history_actions=self._model_config.history_actions,
            future_action_steps=self._prediction.future_action_steps,
            camera_bins=self._training.camera_bins,
            d_model=tcfg.hidden_dim,
            num_layers=tcfg.num_layers,
            num_heads=tcfg.num_heads,
            dropout=tcfg.dropout,
            visual_tokens_per_frame=tcfg.visual_tokens_per_frame,
            memory_slots=self._memory.slots if self._memory.enabled else 0,
            age_decay_action=tcfg.age_decay_action,
            age_decay_visual=tcfg.age_decay_visual,
            age_decay_memory=tcfg.age_decay_memory,
            future_latent_head=tcfg.future_latent_head,
            train_stage=self._training.train_stage,
            pretrained=self._pretrained,
            audio_mels=self._audio_mels(),
        ).to(self._device)

        out_dir = Path(checkpoints_dir) / model_version
        out_dir.mkdir(parents=True, exist_ok=False)
        training_config = self._training_config_snapshot()
        epoch_paths: list[str] = []

        def atomic_state_dict(path: Path) -> None:
            tmp = path.with_name(path.name + ".tmp")
            try:
                torch.save(net.state_dict(), tmp)
                os.replace(tmp, path)
            finally:
                tmp.unlink(missing_ok=True)

        def atomic_meta(meta: ModelCheckpointMeta, path: Path) -> None:
            tmp = path.with_name(path.name + ".tmp")
            try:
                meta.save(tmp)
                os.replace(tmp, path)
            finally:
                tmp.unlink(missing_ok=True)

        def save_epoch(epoch: int, means: dict[str, float]) -> ModelCheckpointMeta:
            relative = f"epochs/epoch-{epoch:03d}"
            epoch_dir = out_dir / relative
            epoch_dir.mkdir(parents=True, exist_ok=False)
            meta = ModelCheckpointMeta(
                model_version=model_version,
                dataset_version=dataset_version,
                code_commit=code_commit,
                training_config=training_config,
                eval_result={"partial": True},
                created_us=now_us(),
                epoch=epoch,
                train_loss={
                    name: means[name]
                    for name in ("move", "camera", "button", "temporal")
                },
                total_loss=means["total"],
                gate={
                    "fast_mean": means["gate_fast_mean"],
                    "fast_std": means["gate_fast_std"],
                    "slow_mean": means["gate_slow_mean"],
                    "slow_std": means["gate_slow_std"],
                },
            )
            atomic_state_dict(epoch_dir / "model.pt")
            atomic_meta(meta, epoch_dir / "meta.json")
            epoch_paths.append(relative)
            return meta

        loader = DataLoader(
            dataset,
            batch_size=self._training.batch_size,
            shuffle=True,
            num_workers=0,  # 帧解码走懒加载，Windows 上多进程 pickle 视频句柄不稳，先单进程
        )
        optimizer = torch.optim.AdamW(
            (p for p in net.parameters() if p.requires_grad), lr=self._training.lr
        )

        # §24：按钮类别不平衡 pos_weight 先全量统计一次
        print(f"[train] 统计按钮 pos_weight（spec §24，先全量扫一遍 {len(dataset)} 个样本）…")
        t_pos = time.time()
        all_buttons = torch.cat([batch["buttons"] for batch in loader], dim=0)
        pos_weight = compute_button_pos_weight(all_buttons).to(self._device)
        print(f"[train] pos_weight 完成（{time.time() - t_pos:.1f}s）")

        batches_per_epoch = len(loader)
        print(
            f"[train] 开始训练: samples={len(dataset)} batch_size={self._training.batch_size} "
            f"batches/epoch={batches_per_epoch} epochs={self._training.epochs} "
            f"device={self._device} audio={'on' if self._audio_mels() else 'off'}"
        )

        history: list[dict[str, float]] = []
        net.train()
        for epoch in range(self._training.epochs):
            epoch_start = time.time()
            last_log = epoch_start
            log_every = max(1, batches_per_epoch // 20)  # 每 epoch ≥5% 一行
            epoch_parts: dict[str, list[float]] = {}
            gate_vals: list[list[float]] = []  # [fast, slow] 逐样本（spec §16 gate 统计）
            for step, batch in enumerate(loader, start=1):
                batch = {k: v.to(self._device) for k, v in batch.items()}
                outputs = _net_forward(net, batch)
                gate_vals.extend(outputs["gates"].detach().cpu().tolist())
                targets = {
                    "move": batch["move"],
                    "camera_bins": batch["camera_bins"],
                    "buttons": batch["buttons"],
                }
                loss, parts = compute_loss(outputs, targets, self._loss_weights, pos_weight)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                for name, value in parts.items():
                    epoch_parts.setdefault(name, []).append(value)
                epoch_parts.setdefault("total", []).append(float(loss.detach()))

                now = time.time()
                if step % log_every == 0 or now - last_log >= 3.0 or step == batches_per_epoch:
                    last_log = now
                    elapsed = now - epoch_start
                    speed = step * self._training.batch_size / max(elapsed, 1e-9)
                    eta_s = elapsed / step * (batches_per_epoch - step)
                    running = {k: float(np.mean(v)) for k, v in epoch_parts.items()}
                    print(
                        f"[train] epoch {epoch + 1}/{self._training.epochs} "
                        f"batch {step}/{batches_per_epoch} ({step * 100 // batches_per_epoch}%) "
                        + " ".join(f"{k}={v:.4f}" for k, v in running.items())
                        + f" | {speed:.1f} samples/s eta {eta_s:.0f}s"
                    )
            epoch_s = time.time() - epoch_start
            means = {name: float(np.mean(vals)) for name, vals in epoch_parts.items()}
            means["epoch"] = float(epoch + 1)
            gate_arr = np.asarray(gate_vals)
            means["gate_fast_mean"] = float(gate_arr[:, 0].mean())
            means["gate_fast_std"] = float(gate_arr[:, 0].std())
            means["gate_slow_mean"] = float(gate_arr[:, 1].mean())
            means["gate_slow_std"] = float(gate_arr[:, 1].std())
            history.append(means)
            print(
                f"[train] epoch {epoch + 1}/{self._training.epochs} done in {epoch_s:.1f}s "
                + " ".join(f"{k}={v:.4f}" for k, v in means.items() if k != "epoch")
            )
            save_epoch(epoch + 1, means)
            print(f"[train] checkpoint 已保存: {out_dir / epoch_paths[-1]}/")

        print("[train] 训练完成，计算训练集指标（spec §36）…")
        if hasattr(dataset, "augment"):
            dataset.augment = False  # 评估/消融不带 Action History 增强
        eval_result = self._evaluate_train_set(net, loader)
        print("[train] 输入依赖消融测试（spec §16：Visual/Action/Memory Dependency）…")
        from evaluation.dependency import dependency_report

        eval_result["dependency"] = dependency_report(net, loader, self._device)
        # spec §16：gate 单极化告警（长期 fast≈1/slow≈0 或反向 = 可能 collapse）
        last_gates = history[-1]
        eval_result["gates"] = {
            "fast_mean": last_gates["gate_fast_mean"],
            "fast_std": last_gates["gate_fast_std"],
            "slow_mean": last_gates["gate_slow_mean"],
            "slow_std": last_gates["gate_slow_std"],
        }
        collapse = (
            last_gates["gate_fast_mean"] > 0.99 and last_gates["gate_slow_mean"] < 0.01
        ) or (
            last_gates["gate_slow_mean"] > 0.99 and last_gates["gate_fast_mean"] < 0.01
        )
        eval_result["possible_gate_collapse"] = bool(collapse)
        if collapse:
            print("[train] 警告: gate 分布单极化，possible_gate_collapse=true（spec §16）")
        selected_epoch = self._training.epochs
        selection_reason = "last_completed_epoch"
        final_dir = out_dir / "final"
        final_dir.mkdir(exist_ok=False)
        final_weights = final_dir / "model.pt"
        final_tmp = final_weights.with_name(final_weights.name + ".tmp")
        try:
            shutil.copy2(out_dir / epoch_paths[-1] / "model.pt", final_tmp)
            os.replace(final_tmp, final_weights)
        finally:
            final_tmp.unlink(missing_ok=True)

        final_meta = ModelCheckpointMeta(
            model_version=model_version,
            dataset_version=dataset_version,
            code_commit=code_commit,
            training_config=training_config,
            eval_result={**eval_result, "loss_history": history},
            created_us=now_us(),
            available_epoch_checkpoints=tuple(epoch_paths),
            selected_epoch=selected_epoch,
            selection_reason=selection_reason,
        )
        atomic_meta(final_meta, final_dir / "meta.json")
        atomic_meta(final_meta, out_dir / "meta.json")
        return final_meta

    @torch.no_grad()
    def _evaluate_train_set(self, net: Any, loader: DataLoader, max_batches: int = 16) -> dict[str, Any]:
        """训练集子集上的 §36 指标（Phase 1 过拟合判据，不是泛化评估）。"""
        from evaluation.offline import evaluate_samples

        net.eval()
        predictions: list[NormalizedAction] = []
        targets: list[NormalizedAction] = []
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            batch = {k: v.to(self._device) for k, v in batch.items()}
            out = _net_forward(net, batch)
            move = out["move"][:, 0].cpu().numpy()
            camera = torch.softmax(out["camera_logits"][:, 0], dim=-1).cpu().numpy()
            buttons = torch.sigmoid(out["button_logits"][:, 0]).cpu().numpy()
            t_move = batch["move"][:, 0].cpu().numpy()
            t_buttons = batch["buttons"][:, 0].cpu().numpy()
            t_camera_bins = batch["camera_bins"][:, 0].cpu().numpy()

            for b in range(move.shape[0]):
                predictions.append(
                    NormalizedAction(
                        move_x=float(move[b, 0]),
                        move_y=float(move[b, 1]),
                        camera_x=bin_probs_to_camera(camera[b, 0]),
                        camera_y=bin_probs_to_camera(camera[b, 1]),
                        buttons=frozenset(
                            name for j, name in enumerate(BUTTONS) if buttons[b, j] > 0.5
                        ),
                    )
                )
                bins = self._training.camera_bins
                targets.append(
                    NormalizedAction(
                        move_x=float(t_move[b, 0]),
                        move_y=float(t_move[b, 1]),
                        camera_x=float(t_camera_bins[b, 0]) / (bins - 1) * 2.0 - 1.0,
                        camera_y=float(t_camera_bins[b, 1]) / (bins - 1) * 2.0 - 1.0,
                        buttons=frozenset(
                            name for j, name in enumerate(BUTTONS) if t_buttons[b, j] > 0.5
                        ),
                    )
                )
        net.train()
        return evaluate_samples(predictions, targets)

    def _training_config_snapshot(self) -> dict[str, Any]:
        """训练配置快照（§29 可复现；含 TorchPolicy 重建结构所需的全部参数）。"""
        from model.torch_model import ARCH_TAG

        tcfg = self._transformer
        return {
            "arch": ARCH_TAG,  # 架构标记：loader 据此拒绝 legacy（GRU/LSTM/早期）checkpoint
            "epochs": self._training.epochs,
            "batch_size": self._training.batch_size,
            "lr": self._training.lr,
            "camera_bins": self._training.camera_bins,
            "train_stage": self._training.train_stage,
            "history_frames": self._model_config.history_frames,
            "history_actions": self._model_config.history_actions,
            "input_width": self._model_config.input_width,
            "input_height": self._model_config.input_height,
            "sample_fps": self._model_config.sample_fps,
            "action_step_ms": self._prediction.action_step_ms,
            "future_action_steps": self._prediction.future_action_steps,
            "execute_steps": self._prediction.execute_steps,
            "transformer": {
                "hidden_dim": tcfg.hidden_dim,
                "num_layers": tcfg.num_layers,
                "num_heads": tcfg.num_heads,
                "dropout": tcfg.dropout,
                "visual_tokens_per_frame": tcfg.visual_tokens_per_frame,
                "future_latent_head": tcfg.future_latent_head,
                "age_decay_action": tcfg.age_decay_action,
                "age_decay_visual": tcfg.age_decay_visual,
                "age_decay_memory": tcfg.age_decay_memory,
            },
            "memory": {
                "enabled": self._memory.enabled,
                "slots": self._memory.slots,
                "update_interval_ms": self._memory.update_interval_ms,
            },
            # spec §8.5：None=无音频分支；加载时按此重建结构与特征参数
            "audio": self._audio_snapshot(),
            "loss_weights": {
                "move": self._loss_weights.move,
                "camera": self._loss_weights.camera,
                "button": self._loss_weights.button,
                "temporal": self._loss_weights.temporal,
            },
        }


def _net_forward(net: Any, batch: dict[str, Any]) -> dict[str, Any]:
    """统一训练/评估前向调用：batch 缺 memory/audio 键时传 None（对应分支关闭）。"""
    return net(
        batch["frames"],
        batch["frame_ages"],
        batch["action_hist"],
        batch["action_ages"],
        memory_frames=batch.get("memory_frames"),
        memory_ages=batch.get("memory_ages"),
        audio_mel=batch.get("audio_mel"),
    )
