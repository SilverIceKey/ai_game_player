"""VideoActionNet：torch 模型本体（spec §16 架构 + §19 拆 Head）。

```text
Video Frames (B,k,3,H,W)          Action History (B,m,18)     Audio (B,mels,T) §8.5 可选
    │                                   │                          │
    ▼ (预训练 ResNet18 去 fc)            ▼ mean-pool                ▼ 小 CNN
Visual Tokens (B,k,512)           Action Vec (B,18)           Audio Vec (B,128)
    │                                   │                          │
    ▼ GRU Temporal Encoder              │                          │
最后隐状态 (B,hidden) ◄──concat─────────┴──────────────────────────┘
    │
    ▼ MLP Policy Decoder → 未来 n 步
    ├─ Movement Head: (B,n,2) tanh           §19.1 regression
    ├─ Camera Head:   (B,n,2,bins) logits    §19.2 discretized distribution
    └─ Button Head:   (B,n,14) logits        §19.3 multi-label binary
```

spec §18 三阶段训练策略（train_stage）：freeze_backbone（默认）→
unfreeze_last（解冻 layer4）→ full。禁止随机初始化视觉系统：
pretrained=True 使用 ImageNet 权重（测试传 pretrained=False 避免下载）。

本模块顶层 import torch/torchvision：只应在 torch 路径上被延迟导入
（load_policy / Trainer / app.train），无 torch 环境只要不 import 本模块即可。
"""
from __future__ import annotations

import torch
import torchvision
from torch import nn

from capture.action import BUTTONS
from model.encoding import ACTION_DIM

# spec §18 三阶段合法值（与 config.TrainingConfig.train_stage 一致）
TRAIN_STAGES = ("freeze_backbone", "unfreeze_last", "full")

NUM_BUTTONS = len(BUTTONS)


class VideoActionNet(nn.Module):
    def __init__(
        self,
        history_frames: int = 16,
        future_action_steps: int = 4,
        camera_bins: int = 21,
        hidden_dim: int = 256,
        train_stage: str = "freeze_backbone",
        pretrained: bool = True,
        audio_mels: int | None = None,  # spec §8.5：音频分支（None=关闭）
        audio_dim: int = 128,
    ):
        super().__init__()
        if train_stage not in TRAIN_STAGES:
            raise ValueError(f"train_stage 非法: {train_stage!r}（合法值 {TRAIN_STAGES}）")
        self.history_frames = history_frames
        self.future_action_steps = future_action_steps
        self.camera_bins = camera_bins
        self.audio_mels = audio_mels

        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = torchvision.models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # (B,512,1,1)

        self.gru = nn.GRU(input_size=512, hidden_size=hidden_dim, batch_first=True)
        if audio_mels is not None:
            # log-mel (B,1,mels,T) → 128 维；时间维自适应池化以容忍窗口长度抖动
            self.audio_cnn = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(32, audio_dim),
                nn.ReLU(),
            )
        decoder_in = hidden_dim + ACTION_DIM + (audio_dim if audio_mels is not None else 0)
        self.decoder = nn.Sequential(
            nn.Linear(decoder_in, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.move_head = nn.Linear(256, future_action_steps * 2)
        self.camera_head = nn.Linear(256, future_action_steps * 2 * camera_bins)
        self.button_head = nn.Linear(256, future_action_steps * NUM_BUTTONS)

        self.apply_train_stage(train_stage)

    def apply_train_stage(self, stage: str) -> None:
        """spec §18：按阶段设置 backbone 可训练范围。"""
        if stage not in TRAIN_STAGES:
            raise ValueError(f"train_stage 非法: {stage!r}（合法值 {TRAIN_STAGES}）")
        for p in self.backbone.parameters():
            p.requires_grad = stage == "full"
        if stage == "unfreeze_last":
            for p in self.backbone[-2].parameters():  # layer4
                p.requires_grad = True

    def forward(
        self,
        frames: torch.Tensor,
        action_hist: torch.Tensor,
        audio_mel: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """frames (B,k,3,H,W) 已归一化；action_hist (B,m,18) 已左 pad；
        audio_mel (B,mels,T) log-mel（音频分支开启时必传，spec §8.5）。"""
        B, k = frames.shape[0], frames.shape[1]
        emb = self.backbone(frames.reshape(B * k, *frames.shape[2:])).reshape(B, k, 512)
        out, _ = self.gru(emb)
        parts = [out[:, -1], action_hist.mean(dim=1)]
        if self.audio_mels is not None:
            if audio_mel is None:
                raise ValueError("模型启用了音频分支（audio_mels），forward 必须传 audio_mel")
            parts.append(self.audio_cnn(audio_mel.unsqueeze(1)))
        context = torch.cat(parts, dim=-1)
        feats = self.decoder(context)
        n = self.future_action_steps
        return {
            "move": torch.tanh(self.move_head(feats)).reshape(B, n, 2),
            "camera_logits": self.camera_head(feats).reshape(B, n, 2, self.camera_bins),
            "button_logits": self.button_head(feats).reshape(B, n, NUM_BUTTONS),
        }
