"""全自动模式入口（CLI）。

用法：python -m apps.auto_player --game wukong --config configs/settings.yaml
"""
from __future__ import annotations

import sys

from apps.auto_player.main import main

if __name__ == "__main__":
    sys.exit(main())
