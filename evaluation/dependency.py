"""输入依赖消融测试（spec §16 修订：Visual / Action / Memory Dependency Test）。

对同一批样本跑 normal 与若干扰动变体，比较 step-0 的 move/camera 误差：

- shuffled_video：样本内帧序打乱（帧与年龄同步置换，内容-年龄配对不变，破坏时序）
- zero_action：Action History 清零 + PAD 年龄（等效无动作历史）
- shuffled_action：Action History 步序打乱
- zero_memory / shuffled_memory：Memory 槽清零 / 打乱（仅 memory 开启时）

解读（spec §19/§20 风险项）：
- shuffled_video 误差几乎不变 → 模型可能没真正用视觉（Visual Ignoring）
- zero_action 误差崩溃式上升 → 可能过度依赖 Action History（Action Shortcut）
- zero_memory 误差几乎不变 → Memory 没学到有效信息（Memory Ignoring）

本模块顶层 import torch：只在训练/评估路径上延迟导入（trainer 收尾调用）。
"""
from __future__ import annotations

from typing import Any

import torch

from model.torch_model import PAD_AGE_S

# (变体名, 是否需要 memory 键)
VARIANTS: tuple[str, ...] = (
    "normal",
    "shuffled_video",
    "zero_action",
    "shuffled_action",
    "zero_memory",
    "shuffled_memory",
)


def _perturb(batch: dict[str, torch.Tensor], variant: str) -> dict[str, torch.Tensor]:
    b = dict(batch)
    if variant == "shuffled_video":
        perm = torch.randperm(b["frames"].shape[1], device=b["frames"].device)
        b["frames"] = b["frames"][:, perm]
        b["frame_ages"] = b["frame_ages"][:, perm]
    elif variant == "zero_action":
        b["action_hist"] = torch.zeros_like(b["action_hist"])
        b["action_ages"] = torch.full_like(b["action_ages"], PAD_AGE_S)
    elif variant == "shuffled_action":
        perm = torch.randperm(b["action_hist"].shape[1], device=b["action_hist"].device)
        b["action_hist"] = b["action_hist"][:, perm]
        b["action_ages"] = b["action_ages"][:, perm]
    elif variant == "zero_memory" and "memory_frames" in b:
        b["memory_frames"] = torch.zeros_like(b["memory_frames"])
        b["memory_ages"] = torch.full_like(b["memory_ages"], PAD_AGE_S)
    elif variant == "shuffled_memory" and "memory_frames" in b:
        perm = torch.randperm(b["memory_frames"].shape[1], device=b["memory_frames"].device)
        b["memory_frames"] = b["memory_frames"][:, perm]
        b["memory_ages"] = b["memory_ages"][:, perm]
    return b


@torch.no_grad()
def dependency_report(
    net: Any, loader: Any, device: torch.device, max_batches: int = 8
) -> dict[str, Any]:
    """逐变体跑前向，输出各变体误差与相对 normal 的 delta（写入 eval_result["dependency"]）。"""
    was_training = net.training
    net.eval()
    sums: dict[str, list[tuple[float, float]]] = {v: [] for v in VARIANTS}
    counts = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        t_move = batch["move"][:, 0]
        t_cam = batch["camera_bins"][:, 0].float()
        for variant in VARIANTS:
            if variant.endswith("memory") and "memory_frames" not in batch:
                continue
            b = _perturb(batch, variant)
            out = net(
                b["frames"],
                b["frame_ages"],
                b["action_hist"],
                b["action_ages"],
                memory_frames=b.get("memory_frames"),
                memory_ages=b.get("memory_ages"),
                audio_mel=b.get("audio_mel"),
            )
            move_err = ((out["move"][:, 0] - t_move) ** 2).mean(dim=-1)  # (B,)
            cam_probs = torch.softmax(out["camera_logits"][:, 0], dim=-1)  # (B,2,bins)
            cam_pred = torch.stack(
                [_bin_expectation(cam_probs[:, 0]), _bin_expectation(cam_probs[:, 1])],
                dim=-1,
            )
            cam_target = t_cam / (cam_probs.shape[-1] - 1) * 2.0 - 1.0
            cam_err = ((cam_pred - cam_target) ** 2).mean(dim=-1)
            sums[variant].extend(zip(move_err.tolist(), cam_err.tolist(), strict=True))
        counts += batch["frames"].shape[0]
    if was_training:
        net.train()

    report: dict[str, Any] = {"sample_count": counts, "variants": {}}
    for variant, rows in sums.items():
        if not rows:
            continue
        report["variants"][variant] = {
            "movement_error": sum(r[0] for r in rows) / len(rows),
            "camera_error": sum(r[1] for r in rows) / len(rows),
        }
    normal = report["variants"].get("normal")
    if normal:
        report["delta_vs_normal"] = {
            name: {
                "movement_error": v["movement_error"] - normal["movement_error"],
                "camera_error": v["camera_error"] - normal["camera_error"],
            }
            for name, v in report["variants"].items()
            if name != "normal"
        }
    return report


def _bin_expectation(probs: torch.Tensor) -> torch.Tensor:
    """bin 概率 → 连续 camera 值（张量版 model.encoding.bin_probs_to_camera）。"""
    bins = probs.shape[-1]
    centers = torch.arange(bins, device=probs.device, dtype=probs.dtype) / (bins - 1) * 2.0 - 1.0
    return (probs * centers).sum(dim=-1)
