"""全自动模式入口（CLI）。

用法：python -m apps.auto_player --game wukong --config configs/settings.yaml
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="全自动游戏 AI（仅截屏 + 模拟输入，不读内存不注入）"
    )
    parser.add_argument("--game", required=True, help="游戏适配器名，如 wukong")
    parser.add_argument("--config", default="configs/settings.yaml", help="配置文件路径")
    args = parser.parse_args()

    print(f"[auto_player] game={args.game} config={args.config}")
    print("[auto_player] M1 链路尚未实现，见 docs/plans/PLAN-20260729-project-skeleton-v1.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
