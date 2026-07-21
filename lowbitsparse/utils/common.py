"""通用工具:随机种子、日志、配置加载、结果落盘、环境信息。"""
import json
import logging
import os
import random
import sys
from datetime import datetime, timezone

import numpy as np


def set_seed(seed: int = 42) -> None:
    """固定随机种子,保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_logger(name: str = "lowbitsparse") -> logging.Logger:
    """返回带统一格式的 logger(避免重复添加 handler)。"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
        handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def load_config(path: str) -> dict:
    """加载 YAML 实验配置。"""
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def env_info() -> dict:
    """收集运行环境信息,写入结果 json 便于复现。"""
    info = {"timestamp": datetime.now(timezone.utc).isoformat()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["gpu"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        )
    except ImportError:
        info["torch"] = None
    return info


def save_results(results: dict, out_dir: str, exp_id: str) -> str:
    """把实验指标 + 环境信息落成 results/<exp_id>.json。"""
    os.makedirs(out_dir, exist_ok=True)
    payload = {"exp_id": exp_id, "env": env_info(), **results}
    path = os.path.join(out_dir, f"{exp_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path
