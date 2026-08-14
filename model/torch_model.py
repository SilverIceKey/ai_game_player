"""VideoActionNet：Token Transformer Temporal Policy（spec §16 v1.0 修订，唯一时序实现）。

```text
Recent Frames (B,k,3,H,W)        Action History (B,m,18)      Memory Frames (B,S,3,H,W)
    │ backbone 去 fc/avgpool         │ Linear                      │ backbone+compressor
    ▼ (B·k,512,⌊H/32⌋,⌊W/32⌋)      ▼                             ▼ MemoryWriter(逐帧)
Visual spatial map              Action Tokens (B,m,d)         Memory Slots (B,S,d)
    │ TokenCompressor（Kt 个 learned query 交叉注意力压缩）
    ▼
Visual Tokens (B,k·Kt,d)
    │
    ▼ 拼接 [VISUAL(k·Kt) | ACTION(m) | MEMORY(S)] + TypeEmbedding
Temporal Transformer（key 侧 age attention bias：-λ_type·age_s，λ 配置化）
    │
    ▼ Learned Gate：z = z_cur + g_fast·z_fast + g_slow·z_slow（连续值，无人工语义）
    ▼ MLP Policy Decoder → 未来 n 步
    ├─ Movement Head: (B,n,2) tanh           §19.1 regression
    ├─ Camera Head:   (B,n,2,bins) logits    §19.2 discretized distribution
    └─ Button Head:   (B,n,14) logits        §19.3 multi-label binary
```

设计要点：
- 时间编码 = key 侧年龄 attention bias（等价 relative temporal encoding）：
  action 快衰减 / visual 中速 / memory 慢衰减，decay 是 prior，attention 仍可学（spec §8.2/§8.3）。
- 空槽（短历史/无早期帧）约定：内容置零、年龄 = PAD_AGE_S，bias 等效屏蔽，
  池化统计时用年龄掩码剔除。
- Memory 由历史帧的 Temporal Representation 压缩得到，不保存 raw frame/action（§8.3）；
  压缩规则训练/推理同一套（本类的 encode_frames + write_memory）。
- checkpoint training_config 带 "arch" 标记；无标记的旧 GRU/LSTM/早期 transformer
  checkpoint 由 loader 拒绝（legacy/unsupported，不 silent fallback）。

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

# checkpoint training_config 的架构标记（loader 据此拒绝 legacy checkpoint）
ARCH_TAG = "token_transformer_v1"

NUM_BUTTONS = len(BUTTONS)

# token 类型下标（TypeEmbedding 顺序，禁止改动：checkpoint 结构依赖）
TYPE_VISUAL = 0
TYPE_ACTION = 1
TYPE_MEMORY = 2
NUM_TOKEN_TYPES = 3

# 空槽/填充 token 的年龄（秒）：age bias 近 -∞，等效注意力屏蔽
PAD_AGE_S = 1e4

VISUAL_TOKEN_GRIDS = {4: (2, 2), 8: (2, 4), 16: (4, 4)}  # 保留：空间网格语义参考

_BACKBONE_DIM = 512  # ResNet18 layer4 输出通道


class TokenCompressor(nn.Module):
    """视觉空间特征 → 每帧 Kt 个 token（learned query 交叉注意力，spec §16）。

    与输入空间分辨率无关（任意 ⌊H/32⌋×⌊W/32⌋），query 序号即空间槽位身份。
    """

    def __init__(self, d_model: int, num_tokens: int, num_heads: int, dropout: float):
        super().__init__()
        self.queries = nn.Parameter(torch.zeros(1, num_tokens, d_model))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.kv_proj = (
            nn.Identity() if _BACKBONE_DIM == d_model else nn.Linear(_BACKBONE_DIM, d_model)
        )
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        """(N, 512, h, w) → (N, Kt, d_model)。"""
        n = feat_map.shape[0]
        tokens = self.kv_proj(feat_map.flatten(2).transpose(1, 2))  # (N, P, d)
        out, _ = self.attn(self.queries.expand(n, -1, -1), tokens, tokens, need_weights=False)
        return out


class MemoryWriter(nn.Module):
    """单帧 Visual Tokens → 1 个 Memory Slot（spec §8.3：压缩而非保存 raw）。"""

    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def forward(self, frame_tokens: torch.Tensor) -> torch.Tensor:
        """(N, Kt, d) → (N, d)。"""
        return self.mlp(frame_tokens.mean(dim=1))


class VideoActionNet(nn.Module):
    def __init__(
        self,
        history_frames: int = 16,
        history_actions: int = 8,
        future_action_steps: int = 4,
        camera_bins: int = 21,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        visual_tokens_per_frame: int = 8,
        memory_slots: int = 0,  # 0 = 无 memory 分支
        age_decay_action: float = 2.0,
        age_decay_visual: float = 0.5,
        age_decay_memory: float = 0.05,
        future_latent_head: bool = False,
        train_stage: str = "freeze_backbone",
        pretrained: bool = True,
        audio_mels: int | None = None,  # spec §8.5：音频分支（None=关闭）
        audio_dim: int = 128,
    ):
        super().__init__()
        if train_stage not in TRAIN_STAGES:
            raise ValueError(f"train_stage 非法: {train_stage!r}（合法值 {TRAIN_STAGES}）")
        if visual_tokens_per_frame not in VISUAL_TOKEN_GRIDS:
            raise ValueError(
                f"visual_tokens_per_frame 非法: {visual_tokens_per_frame}"
                f"（合法值 {tuple(VISUAL_TOKEN_GRIDS)}，spec §16）"
            )
        if future_latent_head:
            raise NotImplementedError(
                "future_latent_head 为预留接口（spec §16）：当前无合理 latent target，未实现"
            )
        if memory_slots < 0:
            raise ValueError(f"memory_slots 必须为非负整数: {memory_slots!r}")
        self.history_frames = history_frames
        self.history_actions = history_actions
        self.future_action_steps = future_action_steps
        self.camera_bins = camera_bins
        self.audio_mels = audio_mels
        self.memory_slots = memory_slots
        self.d_model = d_model
        self.visual_tokens_per_frame = visual_tokens_per_frame

        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = torchvision.models.resnet18(weights=weights)
        # 去 fc + avgpool：保留空间特征图 (512, ⌊H/32⌋, ⌊W/32⌋)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        self.compressor = TokenCompressor(d_model, visual_tokens_per_frame, num_heads, dropout)
        self.memory_writer = MemoryWriter(d_model)
        self.action_proj = nn.Linear(ACTION_DIM, d_model)
        self.type_embed = nn.Embedding(NUM_TOKEN_TYPES, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.num_heads = num_heads

        # 年龄 → attention bias 的衰减率（buffer：随 state_dict 走，不参与训练）
        self.register_buffer(
            "age_decay",
            torch.tensor([age_decay_visual, age_decay_action, age_decay_memory]),
        )

        self.gate_proj = nn.Linear(d_model, 2)  # [g_fast, g_slow]（无 memory 时只用 g_fast）

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
        decoder_in = d_model + (audio_dim if audio_mels is not None else 0)
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
            for p in self.backbone[-1].parameters():  # layer4（[:-2] 后最后一层）
                p.requires_grad = True

    # ---------- 共享压缩路径（训练/推理同一规则，spec §8.3） ----------

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """(N, 3, H, W) 已归一化帧 → (N, Kt, d_model) Visual Tokens。"""
        return self.compressor(self.backbone(frames))

    def write_memory(self, frame_tokens: torch.Tensor) -> torch.Tensor:
        """(N, Kt, d) → (N, d)：每帧压缩为 1 个 Memory Slot。"""
        return self.memory_writer(frame_tokens)

    # ---------- 前向 ----------

    def forward(
        self,
        frames: torch.Tensor,
        frame_ages: torch.Tensor,
        action_hist: torch.Tensor,
        action_ages: torch.Tensor,
        memory_frames: torch.Tensor | None = None,
        memory_ages: torch.Tensor | None = None,
        audio_mel: torch.Tensor | None = None,
        memory_tokens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """frames (B,k,3,H,W) 已归一化，frame_ages (B,k) 秒；
        action_hist (B,m,18) 已左 pad，action_ages (B,m) 秒（pad=PAD_AGE_S）；
        memory：二选一——memory_frames (B,S,3,H,W) 原始帧（训练路径，梯度进
        Compressor/Writer），或 memory_tokens (B,S,d) 预压缩槽位（推理路径，
        来自 TorchPolicy 的 runtime memory deque）；memory_ages (B,S) 秒；
        audio_mel (B,mels,T) log-mel（音频分支开启时必传，spec §8.5）。

        返回三头输出 + gates (B,2)（[fast, slow]，诊断用，spec §16 gate 统计）。
        """
        B, k = frames.shape[0], frames.shape[1]
        kt = self.visual_tokens_per_frame
        v_tokens = self.encode_frames(frames.reshape(B * k, *frames.shape[2:]))
        return self.forward_tokens(
            v_tokens.reshape(B, k, kt, self.d_model),
            frame_ages,
            action_hist,
            action_ages,
            memory_frames=memory_frames,
            memory_ages=memory_ages,
            audio_mel=audio_mel,
            memory_tokens=memory_tokens,
        )

    def forward_tokens(
        self,
        visual_tokens: torch.Tensor,
        frame_ages: torch.Tensor,
        action_hist: torch.Tensor,
        action_ages: torch.Tensor,
        memory_frames: torch.Tensor | None = None,
        memory_ages: torch.Tensor | None = None,
        audio_mel: torch.Tensor | None = None,
        memory_tokens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """消费已编码 Visual Tokens；runtime cache 与 `forward` 共用此数学路径。"""
        B, k, kt, d = visual_tokens.shape
        if kt != self.visual_tokens_per_frame or d != self.d_model:
            raise ValueError(
                f"visual_tokens shape 非法: {tuple(visual_tokens.shape)}，"
                f"期望 (B,k,{self.visual_tokens_per_frame},{self.d_model})"
            )
        v_tokens = visual_tokens.reshape(B, k * kt, d)  # 帧序 × 帧内槽位序

        a_tokens = self.action_proj(action_hist)  # (B, m, d)

        parts = [v_tokens, a_tokens]
        ages = [frame_ages.repeat_interleave(kt, dim=1), action_ages]  # key 年龄（秒）
        types = [TYPE_VISUAL, TYPE_ACTION]
        if self.memory_slots > 0:
            if memory_ages is None or (memory_frames is None and memory_tokens is None):
                raise ValueError(
                    "模型启用了 memory_slots，forward 必须传 memory_ages "
                    "以及 memory_frames（训练）或 memory_tokens（推理）之一"
                )
            if memory_tokens is None:
                memory_tokens = self.write_memory(
                    self.encode_frames(
                        memory_frames.reshape(B * self.memory_slots, *memory_frames.shape[2:])
                    )
                ).reshape(B, self.memory_slots, self.d_model)
            parts.append(memory_tokens)
            ages.append(memory_ages)
            types.append(TYPE_MEMORY)

        tokens = torch.cat(parts, dim=1)  # (B, L, d)
        type_ids = torch.cat(
            [
                torch.full((p.shape[1],), t, dtype=torch.long, device=tokens.device)
                for p, t in zip(parts, types, strict=True)
            ]
        )
        tokens = tokens + self.type_embed(type_ids).unsqueeze(0)

        # key 侧年龄 attention bias：mask[b,i,j] = -λ_type(j)·age_j（与 query i 无关）
        key_ages = torch.cat(ages, dim=1)  # (B, L)
        key_types = type_ids.expand(B, -1)  # (B, L)
        bias = -self.age_decay[key_types] * key_ages  # (B, L)
        length = tokens.shape[1]
        mask = (
            bias.unsqueeze(1)
            .expand(B, length, length)
            .repeat_interleave(self.num_heads, dim=0)
            .contiguous()
        )  # (B·heads, L, L)

        out = self._encode_temporal(tokens, mask)  # (B, L, d)

        valid = key_ages < PAD_AGE_S  # 空槽不参与池化
        z_cur = _masked_mean(out, valid)
        fast_end = k * kt + self.history_actions
        z_fast = _masked_mean(out[:, :fast_end], valid[:, :fast_end])
        gates = torch.sigmoid(self.gate_proj(z_cur))  # (B, 2)
        z = z_cur + gates[:, 0:1] * z_fast
        if self.memory_slots > 0:
            z_slow = _masked_mean(out[:, fast_end:], valid[:, fast_end:])
            z = z + gates[:, 1:2] * z_slow

        if self.audio_mels is not None:
            if audio_mel is None:
                raise ValueError("模型启用了音频分支（audio_mels），forward 必须传 audio_mel")
            z = torch.cat([z, self.audio_cnn(audio_mel.unsqueeze(1))], dim=-1)

        feats = self.decoder(z)
        n = self.future_action_steps
        return {
            "move": torch.tanh(self.move_head(feats)).reshape(B, n, 2),
            "camera_logits": self.camera_head(feats).reshape(B, n, 2, self.camera_bins),
            "button_logits": self.button_head(feats).reshape(B, n, NUM_BUTTONS),
            "gates": gates,
        }

    def _encode_temporal(
        self, tokens: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """与 norm-first TransformerEncoder 相同的非 fused 路径。

        PyTorch CPU eval fastpath 会把当前 per-head float attention mask 算成 NaN；
        显式走各层公开模块与训练路径数学等价，且不改变任何参数/state_dict key。
        """
        out = tokens
        for layer in self.encoder.layers:
            if not layer.norm_first:
                raise RuntimeError("当前架构要求 norm_first=True")
            normed = layer.norm1(out)
            attended = layer.self_attn(
                normed,
                normed,
                normed,
                attn_mask=mask,
                need_weights=False,
            )[0]
            out = out + layer.dropout1(attended)
            normed = layer.norm2(out)
            feedforward = layer.linear2(
                layer.dropout(layer.activation(layer.linear1(normed)))
            )
            out = out + layer.dropout2(feedforward)
        return self.encoder.norm(out) if self.encoder.norm is not None else out


def _masked_mean(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """(B, L, d) 在 valid (B, L) 上求均值；全无效时退化为全零（不 NaN）。"""
    weights = valid.to(x.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp(min=1.0)
    return (x * weights).sum(dim=1) / denom
