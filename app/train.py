"""训练入口（spec §42 Phase 1 tiny overfit / §7 模型生命周期）。

```text
sessions/ → SessionDataset → Trainer → checkpoints/<model_version>/
                                        └→ ModelRegistry.register_candidate
```

用法（Windows 游戏机，先装 CUDA 版 torch，见 README 训练章节）：

```bash
python -m app.train                          # 用 configs/settings.yaml 的 training 段
python -m app.train --epochs 20 --lr 0.0005  # 覆盖超参
```

spec §6：训练是离线命令，与采集/推理进程完全分离；禁止嵌入 OBSERVE_TRAIN。
训练后模型只是 Candidate（§7），promote 由 registry 流程显式触发，
AUTOPILOT 用 --checkpoint 显式指定，不存在静默覆盖。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from app.common import resolve_settings_path
from config import ConfigError, load_settings

_PROG = "train"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=_PROG,
        description="候选模型训练（spec §23/§42 Phase 1）：从 sessions/ 的 Episode Store "
        "构造样本并训练，产物写入 checkpoints/<model_version>/ 并注册为 candidate。",
    )
    parser.add_argument("--config", default="configs/settings.yaml", help="全局配置文件路径")
    parser.add_argument("--sessions", default=None, help="sessions 根目录（缺省用配置 sessions_dir）")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖 training.epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="覆盖 training.batch_size")
    parser.add_argument("--lr", type=float, default=None, help="覆盖 training.lr")
    parser.add_argument("--model-version", default=None,
                        help="模型版本号（缺省自动递增 model-vNNN）")
    parser.add_argument("--checkpoints-dir", default="checkpoints/", help="checkpoint 输出根目录")
    parser.add_argument("--registry", default="checkpoints/registry.json", help="Model Registry 路径")
    return parser.parse_args(argv)


def _code_commit() -> str:
    """当前 git commit（spec §29 可复现性）；非 git 环境记空串。"""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _next_model_version(checkpoints_dir: Path) -> str:
    """model-vNNN 自动递增（与 dataset versioning 同风格）。"""
    existing = [
        p.name for p in checkpoints_dir.iterdir()
        if p.is_dir() and p.name.startswith("model-v")
    ] if checkpoints_dir.is_dir() else []
    highest = 0
    for name in existing:
        suffix = name.removeprefix("model-v")
        if suffix.isdigit():
            highest = max(highest, int(suffix))
    return f"model-v{highest + 1:03d}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    settings_path = resolve_settings_path(args.config, _PROG)
    try:
        settings = load_settings(settings_path)
    except ConfigError as exc:
        raise SystemExit(f"[{_PROG}] 配置错误: {exc}") from exc

    from dataclasses import replace

    training = settings.training
    if args.epochs is not None:
        training = replace(training, epochs=args.epochs)
    if args.batch_size is not None:
        training = replace(training, batch_size=args.batch_size)
    if args.lr is not None:
        training = replace(training, lr=args.lr)

    try:
        from train.dataset import SessionDataset, build_sample_params, find_session_dirs
        from train.trainer import Trainer
    except RuntimeError as exc:
        raise SystemExit(f"[{_PROG}] 启动失败: {exc}") from exc

    sessions_dir = Path(args.sessions) if args.sessions else Path(settings.sessions_dir)
    try:
        session_dirs = find_session_dirs(sessions_dir)
        dataset = SessionDataset(
            session_dirs,
            build_sample_params(settings),
            camera_bins=training.camera_bins,
            input_width=settings.model.input_width,
            input_height=settings.model.input_height,
            audio=settings.audio,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"[{_PROG}] 数据错误: {exc}") from exc

    checkpoints_dir = Path(args.checkpoints_dir)
    model_version = args.model_version or _next_model_version(checkpoints_dir)
    import torch

    print(
        f"[{_PROG}] sessions={len(session_dirs)} samples={len(dataset)} "
        f"model_version={model_version} device={'cuda' if torch.cuda.is_available() else 'cpu'}"
    )

    trainer = Trainer(
        settings.loss_weights,
        training,
        settings.model,
        settings.prediction,
        audio=settings.audio,
    )
    try:
        meta = trainer.train_candidate(
            dataset,
            dataset_version=settings.dataset_version,
            model_version=model_version,
            code_commit=_code_commit(),
            checkpoints_dir=checkpoints_dir,
        )
    finally:
        dataset.close()

    from train.registry import ModelRegistry

    registry = ModelRegistry.load(args.registry)
    registry.register_candidate(meta)

    print(f"[{_PROG}] checkpoint 已保存: {checkpoints_dir / model_version}/")
    print(f"[{_PROG}] 已注册为 candidate（spec §7）：{args.registry}")
    print(f"[{_PROG}] 训练集指标（Phase 1 过拟合判据，spec §36）：")
    for key, value in meta.eval_result.items():
        if key == "loss_history":
            first, last = value[0], value[-1]
            print(f"  loss: {first['total']:.4f} → {last['total']:.4f}")
        elif isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print(
        f"[{_PROG}] 判据（spec §42 Phase 1）：loss 显著下降且按钮 P/R 明显高于随机 "
        f"→ pipeline 正常；否则按 spec §17 排查顺序（Timestamp → Labels → Dataset → …）检查"
    )
    print(f"[{_PROG}] 下一步：python -m app.autopilot --game {settings.game} "
          f"--checkpoint {checkpoints_dir / model_version} --dry-run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
