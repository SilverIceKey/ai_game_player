"""TorchPolicy：加载训练好的 VideoActionNet，实现 VideoActionPolicy 推理协议。

推理侧解码（与 train/losses.py 的训练目标对应）：
- move：tanh 输出直接用（已是 [-1,1]）
- camera：bin softmax 后取期望（比 argmax 平滑，spec §19.2）
- buttons：sigmoid > 0.5 判按下；confidence 记录各按钮概率（spec §33 日志字段）

帧输入是 runtime/preprocess.py 的输出（384×216 HWC float[0,1]，BGR），
归一化与训练侧共用 model/encoding.py，保证 train/inference 一致。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from capture.action import BUTTONS, ActionChunk, NormalizedAction
from capture.clock import now_us
from model.checkpoint import ModelCheckpointMeta
from model.encoding import action_to_vector, bin_probs_to_camera, normalize_frame
from model.torch_model import VideoActionNet


class TorchPolicy:
    """实现 model/policy.py 的 VideoActionPolicy 协议。"""

    def __init__(
        self,
        net: VideoActionNet,
        meta: ModelCheckpointMeta,
        device: torch.device | None = None,
    ):
        self._net = net
        self._meta = meta
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net.to(self._device).eval()

    @property
    def model_version(self) -> str:
        return self._meta.model_version

    @property
    def needs_audio(self) -> bool:
        """checkpoint 是否带音频分支（spec §8.5）；是则运行时必须供给音频。"""
        return self._net.audio_mels is not None

    @torch.no_grad()
    def predict(
        self,
        frames: list[np.ndarray],
        action_history: list[NormalizedAction],
        audio_pcm: np.ndarray | None = None,
    ) -> ActionChunk:
        k = self._net.history_frames
        if len(frames) < k:
            raise ValueError(f"Video History 不足：需要 {k} 帧，实际 {len(frames)}")
        frames = frames[-k:]

        frame_tensor = torch.from_numpy(
            np.stack([normalize_frame(f) for f in frames])
        ).unsqueeze(0).to(self._device)

        hist = np.zeros((1, 1, 18), dtype=np.float32)  # 无历史时全零（与训练左 pad 一致）
        if action_history:
            hist = np.stack([action_to_vector(a) for a in action_history])[None, :, :]
        hist_tensor = torch.from_numpy(np.ascontiguousarray(hist, dtype=np.float32)).to(self._device)

        audio_tensor = None
        if self._net.audio_mels is not None:
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

        out = self._net(frame_tensor, hist_tensor, audio_tensor)
        move = out["move"][0].cpu().numpy()  # (n, 2)
        camera = torch.softmax(out["camera_logits"][0], dim=-1).cpu().numpy()  # (n, 2, bins)
        button_probs = torch.sigmoid(out["button_logits"][0]).cpu().numpy()  # (n, 14)

        actions = []
        for step in range(self._net.future_action_steps):
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

        return ActionChunk(
            actions=tuple(actions),
            step_ms=float(self._meta.training_config.get("action_step_ms", 50.0)),
            model_version=self.model_version,
            confidence={
                name: float(button_probs[0, i]) for i, name in enumerate(BUTTONS)
            },
            created_us=now_us(),
        )


def load_torch_policy(checkpoint_dir: str | Path) -> TorchPolicy:
    """从 checkpoints/<version>/ 加载：meta.json 描述结构参数，model.pt 为权重。"""
    directory = Path(checkpoint_dir)
    if directory.is_file():  # 容忍直接传 meta.json / model.pt 文件路径
        directory = directory.parent
    meta = ModelCheckpointMeta.load(directory / "meta.json")
    cfg = meta.training_config

    audio_cfg = cfg.get("audio")  # spec §8.5：None=无音频分支
    net = VideoActionNet(
        history_frames=int(cfg.get("history_frames", 16)),
        future_action_steps=int(cfg.get("future_action_steps", 4)),
        camera_bins=int(cfg.get("camera_bins", 21)),
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        train_stage=str(cfg.get("train_stage", "freeze_backbone")),
        pretrained=False,  # 加载训练权重，不需要 ImageNet 预训练
        audio_mels=int(audio_cfg["mels"]) if audio_cfg else None,
    )
    weights_path = directory / "model.pt"
    if not weights_path.is_file():
        raise FileNotFoundError(f"checkpoint 权重缺失: {weights_path}")
    net.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    return TorchPolicy(net, meta)
