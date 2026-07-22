"""测试入口路径修正。

保证直接运行 `pytest` 时也能从仓库根目录导入 `lowbitsparse`。
"""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
