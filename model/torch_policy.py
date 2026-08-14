"""TorchPolicy：加载训练好的 VideoActionNet，实现 VideoActionPolicy 推理协议。

推理侧解码（与 train/losses.py 的训练目标对应）：
- move：tanh 输出直接用（已是 [-1,1]）
- camera：bin softmax 后取期望（比 argmax 平滑，spec §19.2）
- buttons：sigmoid > 0.5 判按下；confidence 记录各按钮概率（spec §33 日志字段）

帧输入是 runtime/preprocess.py 的输出（384×216 HWC float[0,1]，BGR）配时间戳，
归一化与训练侧共用 model/encoding.py，保证 train/inference 一致。

Runtime Memory（spec §8.3）：memory 是推理侧运行状态，不进 checkpoint。
Visual Token cache 按 frame timestamp 仅编码新帧；每次 predict 直接把最近 K 帧
缓存 token 交给 forward_tokens。MemoryWriter 复用最新帧缓存 token 压 1 个 slot，
不重复调用 backbone/compressor。
reset_memory() 是 Hard Reset 入口（死亡/读档/新游戏等由 app 层判定触发）。
"""
from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from capture.action import BUTTONS, ActionChunk, ActionRecord, NormalizedAction
from capture.clock import now_us
from model.checkpoint import ModelCheckpointMeta
from model.encoding import action_to_vector, bin_probs_to_camera, normalize_frame
from model.torch_model import ARCH_TAG, PAD_AGE_S, VideoActionNet


class TorchPolicy:
    """实现 model/policy.py 的 VideoActionPolicy 协议。"""

    def __init__(
        self,
        net: VideoActionNet,
        meta: ModelCheckpointMeta,
        device: torch.device | None = None,
        fp16_autocast: bool = False,
    ):
        self._net = net
        self._meta = meta
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net.to(self._device).eval()
        self._fp16_autocast = bool(fp16_autocast and self._device.type == "cuda")
        self._visual_tokens: deque[tuple[int, torch.Tensor]] = deque(
            maxlen=net.history_frames
        )
        mem_cfg = meta.training_config.get("memory") or {}
        self._mem_interval_us = int(mem_cfg.get("update_interval_ms", 500)) * 1000
        self._mem_slots: deque[tuple[torch.Tensor, int]] = deque(
            maxlen=max(1, net.memory_slots)
        )
        self._mem_last_write_us: int | None = None
        self._mem_updates = 0
        self._mem_resets = 0
        self.last_diagnostics: dict[str, float] = {}

    @property
    def model_version(self) -> str:
        return self._meta.model_version

    @property
    def needs_audio(self) -> bool:
        """checkpoint 是否带音频分支（spec §8.5）；是则运行时必须供给音频。"""
        return self._net.audio_mels is not None

    def reset_memory(self) -> None:
        """Hard Reset（spec §8.3）：清空 Memory Slots，由 app 层在死亡/读档/新游戏时调用。"""
        self._mem_slots.clear()
        self._mem_last_write_us = None
        self._mem_resets += 1

    @torch.inference_mode()
    def predict(
        self,
        frames: list[tuple[np.ndarray, int]],
        action_history: list[ActionRecord],
        audio_pcm: np.ndarray | None = None,
    ) -> ActionChunk:
        """frames 为 (frame, timestamp_us) 列表（旧→新），action_history 为 ActionRecord 列表。"""
        net = self._net
        k = net.history_frames
        if len(frames) < k:
            raise ValueError(f"Video History 不足：需要 {k} 帧，实际 {len(frames)}")
        window = frames[-k:]
        anchor_us = int(window[-1][1])  # 最新帧时刻 = 推理 anchor

        frame_ages = torch.tensor(
            [[(anchor_us - int(ts)) / 1e6 for _, ts in window]],
            dtype=torch.float32,
            device=self._device,
        )

        # Action History 左 pad（与训练侧 _left_pad_history 同规则）
        m = net.history_actions
        hist = np.zeros((m, 18), dtype=np.float32)
        ages = np.full(m, PAD_AGE_S, dtype=np.float32)
        recent = action_history[-m:]
        for i, record in enumerate(recent):
            pos = m - len(recent) + i
            hist[pos] = action_to_vector(record.action)
            ages[pos] = (anchor_us - record.timestamp_us) / 1e6
        hist_tensor = torch.from_numpy(hist).unsqueeze(0).to(self._device)
        age_tensor = torch.from_numpy(ages).unsqueeze(0).to(self._device)

        audio_tensor = None
        if net.audio_mels is not None:
            if audio_pcm is None:
                raise ValueError("模型启用了音频分支，predict 必须传 audio_pcm（spec §8.5）")
            from model.audio_features import log_mel

            cfg = self._meta.training_config["audio"]  # 训练时快照的特征参数
            mel = log_mel(
                np.asarray(audio_pcm, dtype=np.float32),
                sample_rate=int(cfg["sample_rate"]),
                mels=int(cfg["mels"]),
                fft_size=int(cfg["fft_size"]),
                hop_size=int(cfg["hop_size"]),
            )
            audio_tensor = torch.from_numpy(mel).unsqueeze(0).to(self._device)

        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self._fp16_autocast
            else nullcontext()
        )
        with autocast:
            visual_start = self._timer_start()
            visual_tokens = self._visual_window(window)
            visual_mark = self._timer_end(visual_start)

            memory_tokens = None
            memory_ages = None
            memory_mark: Any = 0.0
            memory_written = False
            if net.memory_slots > 0:
                memory_start = self._timer_start()
                memory_written = self._maybe_write_memory(
                    visual_tokens[:, -1], anchor_us
                )
                memory_mark = self._timer_end(memory_start)
                memory_tokens, memory_ages = self._memory_tensors(anchor_us)

            transformer_start = self._timer_start()
            out = net.forward_tokens(
                visual_tokens,
                frame_ages,
                hist_tensor,
                age_tensor,
                memory_ages=memory_ages,
                audio_mel=audio_tensor,
                memory_tokens=memory_tokens,
            )
            transformer_mark = self._timer_end(transformer_start)

        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        visual_encode_ms = self._timer_ms(visual_mark)
        transformer_ms = self._timer_ms(transformer_mark)
        memory_write_ms = self._timer_ms(memory_mark) if memory_written else 0.0
        if not all(bool(torch.isfinite(value).all()) for value in out.values()):
            raise RuntimeError("模型推理输出含 NaN/Inf，拒绝下发动作")

        decode_start = perf_counter()
        gates = out["gates"][0].cpu()
        move = out["move"][0].cpu().numpy()  # (n, 2)
        camera = torch.softmax(out["camera_logits"][0], dim=-1).cpu().numpy()  # (n, 2, bins)
        button_probs = torch.sigmoid(out["button_logits"][0]).cpu().numpy()  # (n, 14)

        actions = []
        for step in range(net.future_action_steps):
            buttons = frozenset(
                name for i, name in enumerate(BUTTONS) if button_probs[step, i] > 0.5
            )
            actions.append(
                NormalizedAction(
                    move_x=float(move[step, 0]),
                    move_y=float(move[step, 1]),
                    camera_x=bin_probs_to_camera(camera[step, 0]),
                    camera_y=bin_probs_to_camera(camera[step, 1]),
                    buttons=buttons,
                )
            )

        chunk = ActionChunk(
            actions=tuple(actions),
            step_ms=float(self._meta.training_config.get("action_step_ms", 50.0)),
            model_version=self.model_version,
            confidence={
                name: float(button_probs[0, i]) for i, name in enumerate(BUTTONS)
            },
            created_us=now_us(),
        )
        decode_ms = (perf_counter() - decode_start) * 1000.0
        self.last_diagnostics = {
            "fast_gate": float(gates[0]),
            "memory_gate": float(gates[1]),
            "visual_tokens": float(k * net.visual_tokens_per_frame),
            "action_tokens": float(m),
            "memory_tokens": float(net.memory_slots),
            "memory_slots_filled": float(len(self._mem_slots)),
            "memory_updates": float(self._mem_updates),
            "memory_resets": float(self._mem_resets),
            "visual_encode_ms": visual_encode_ms,
            "transformer_ms": transformer_ms,
            "memory_write_ms": memory_write_ms,
            "decode_ms": decode_ms,
            "fp16_autocast": float(self._fp16_autocast),
        }
        return chunk

    def _visual_window(
        self, window: list[tuple[np.ndarray, int]]
    ) -> torch.Tensor:
        """只编码未缓存帧，返回 (1,k,Kt,d) Visual Tokens。"""
        timestamps = [int(ts) for _, ts in window]
        if len(set(timestamps)) != len(timestamps):
            raise ValueError("Video History 含重复 timestamp，无法可靠识别 Visual Token cache")
        cached = dict(self._visual_tokens)
        missing = [(frame, ts) for frame, ts in window if int(ts) not in cached]
        if missing:
            tensor = torch.from_numpy(
                np.stack([normalize_frame(frame) for frame, _ in missing])
            ).to(self._device)
            encoded = self._net.encode_frames(tensor)
            for (_, ts), tokens in zip(missing, encoded, strict=True):
                self._visual_tokens.append((int(ts), tokens))
            cached = dict(self._visual_tokens)
        return torch.stack([cached[ts] for ts in timestamps]).unsqueeze(0)

    def _maybe_write_memory(self, latest_tokens: torch.Tensor, anchor_us: int) -> bool:
        """距上次写入 ≥ interval 时，复用最新帧 Visual Tokens 写 1 个 slot。"""
        if (
            self._mem_last_write_us is not None
            and anchor_us - self._mem_last_write_us < self._mem_interval_us
        ):
            return False
        slot = self._net.write_memory(latest_tokens)[0]
        self._mem_slots.append((slot.cpu(), anchor_us))
        self._mem_last_write_us = anchor_us
        self._mem_updates += 1
        return True

    def _timer_start(self) -> float | torch.cuda.Event:
        if self._device.type != "cuda":
            return perf_counter()
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def _timer_end(
        self, start: float | torch.cuda.Event
    ) -> float | tuple[torch.cuda.Event, torch.cuda.Event]:
        if isinstance(start, float):
            return (perf_counter() - start) * 1000.0
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        return start, end

    @staticmethod
    def _timer_ms(mark: Any) -> float:
        if isinstance(mark, float):
            return mark
        return float(mark[0].elapsed_time(mark[1]))

    def _memory_tensors(self, anchor_us: int) -> tuple[torch.Tensor, torch.Tensor]:
        """memory deque → (1,S,d) tokens + (1,S) 年龄秒（空槽零向量 + PAD_AGE_S）。"""
        net = self._net
        slots = torch.zeros(1, net.memory_slots, net.d_model)
        ages = np.full(net.memory_slots, PAD_AGE_S, dtype=np.float32)
        for j, (slot, ts) in enumerate(self._mem_slots):
            slots[0, j] = slot
            ages[j] = (anchor_us - ts) / 1e6
        return (
            slots.to(self._device),
            torch.from_numpy(ages).unsqueeze(0).to(self._device),
        )


def load_torch_policy(
    checkpoint_dir: str | Path, fp16_autocast: bool = False
) -> TorchPolicy:
    """加载 model 根目录、final/、epoch-NNN/ 或旧版平铺 checkpoint。

    无 "arch": ARCH_TAG 标记的 checkpoint（GRU/LSTM/早期 transformer）
    一律拒绝：legacy / unsupported，不做 silent fallback（spec §16）。
    """
    directory = Path(checkpoint_dir)
    if directory.is_file():  # 容忍直接传 meta.json / model.pt 文件路径
        directory = directory.parent
    if (directory / "final").is_dir():
        directory = directory / "final"
    elif not (directory / "model.pt").is_file() and (directory / "meta.json").is_file():
        summary = ModelCheckpointMeta.load(directory / "meta.json")
        if not summary.available_epoch_checkpoints:
            raise FileNotFoundError(f"checkpoint 权重缺失: {directory / 'model.pt'}")
        available = ", ".join(summary.available_epoch_checkpoints) or "无"
        raise FileNotFoundError(
            f"checkpoint 尚未生成 final/: {directory}；已完成 epoch: {available}。"
            "请显式传入 epochs/epoch-NNN"
        )
    meta = ModelCheckpointMeta.load(directory / "meta.json")
    cfg = meta.training_config

    if cfg.get("arch") != ARCH_TAG:
        raise RuntimeError(
            f"checkpoint {directory} 是 legacy/unsupported 架构（training_config.arch="
            f"{cfg.get('arch')!r}，当前仅支持 {ARCH_TAG!r}）：GRU/LSTM 已移除，"
            "请用当前代码重新训练"
        )

    tcfg = cfg.get("transformer") or {}
    mem_cfg = cfg.get("memory") or {}
    audio_cfg = cfg.get("audio")  # spec §8.5：None=无音频分支
    net = VideoActionNet(
        history_frames=int(cfg.get("history_frames", 16)),
        history_actions=int(cfg.get("history_actions", 8)),
        future_action_steps=int(cfg.get("future_action_steps", 4)),
        camera_bins=int(cfg.get("camera_bins", 21)),
        d_model=int(tcfg.get("hidden_dim", 512)),
        num_layers=int(tcfg.get("num_layers", 6)),
        num_heads=int(tcfg.get("num_heads", 8)),
        dropout=float(tcfg.get("dropout", 0.1)),
        visual_tokens_per_frame=int(tcfg.get("visual_tokens_per_frame", 8)),
        memory_slots=int(mem_cfg.get("slots", 0)) if mem_cfg.get("enabled") else 0,
        age_decay_action=float(tcfg.get("age_decay_action", 2.0)),
        age_decay_visual=float(tcfg.get("age_decay_visual", 0.5)),
        age_decay_memory=float(tcfg.get("age_decay_memory", 0.05)),
        train_stage=str(cfg.get("train_stage", "freeze_backbone")),
        pretrained=False,  # 加载训练权重，不需要 ImageNet 预训练
        audio_mels=int(audio_cfg["mels"]) if audio_cfg else None,
    )
    weights_path = directory / "model.pt"
    if not weights_path.is_file():
        raise FileNotFoundError(f"checkpoint 权重缺失: {weights_path}")
    net.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    return TorchPolicy(net, meta, fp16_autocast=fp16_autocast)
