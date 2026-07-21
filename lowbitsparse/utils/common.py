"""通用工具模块。

集中放置与具体压缩算法无关的基础设施:随机种子、日志、YAML 配置加载、
实验结果落盘、运行环境采集。所有里程碑(M0-M4)共用。

设计约定:
- 顶层不 import torch —— 本地(无 GPU / 无 torch)也要能导入本模块做接线测试,
  torch 仅在需要处于函数内部延迟导入,导入失败则静默降级。
"""
import json          # 实验结果序列化为 json
import logging       # 统一日志输出
import os            # 路径拼接 / 目录创建
import random        # Python 内置随机源
import sys           # 日志写到 stdout(Colab 可见)
from datetime import datetime, timezone  # 生成带时区的 UTC 时间戳

import numpy as np   # numpy 随机源(不依赖 GPU,可放顶层)


def set_seed(seed: int = 42) -> None:
    """固定所有随机源,保证实验可复现。

    参数:
        seed: 随机种子,默认 42;全项目统一用同一 seed 便于横向对比。

    覆盖 Python random、numpy、torch(含所有 CUDA 设备)三处随机源。
    torch 用 try/except 延迟导入,本地无 torch 时跳过而非报错。
    """
    random.seed(seed)          # Python 层随机(如数据打乱)
    np.random.seed(seed)       # numpy 随机(如校准集采样)
    try:
        import torch
        torch.manual_seed(seed)            # CPU 端随机(初始化 / dropout)
        torch.cuda.manual_seed_all(seed)   # 所有 GPU 端随机
    except ImportError:
        pass  # 无 torch 的本地接线测试:静默跳过


def get_logger(name: str = "lowbitsparse") -> logging.Logger:
    """返回带统一格式的 logger,并避免重复添加 handler。

    参数:
        name: logger 名称,默认 "lowbitsparse";同名 logger 全局复用同一实例。
    返回:
        配置好的 logging.Logger,输出形如 [12:00:00] INFO lowbitsparse: 消息。

    多次调用(如各子模块各拿一次)只会在首次初始化 handler,
    靠 `if not logger.handlers` 幂等判断,防止日志重复打印。
    """
    logger = logging.getLogger(name)         # 按名取全局单例
    if not logger.handlers:                  # 仅首次初始化,幂等
        handler = logging.StreamHandler(sys.stdout)      # 输出到 stdout
        fmt = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
        handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)        # 默认 INFO 级别
        logger.propagate = False             # 不向 root 冒泡,避免二次输出
    return logger


def load_config(path: str) -> dict:
    """加载 YAML 实验配置为 dict。

    参数:
        path: YAML 文件路径,如 configs/qwen0.5b_base.yaml。
    返回:
        解析后的配置字典;空文件返回 {} 而非 None,便于下游 .get() 取默认值。

    yaml 在函数内延迟导入,保持顶层零重依赖。
    """
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        # safe_load 只解析基础类型,拒绝执行任意标签,避免恶意 YAML 风险
        return yaml.safe_load(f) or {}


def env_info() -> dict:
    """采集运行环境信息,嵌入结果 json 以保证实验可复现。

    返回:
        dict,含:
            timestamp: UTC ISO 时间戳(用 timezone.utc 避免本地时区歧义);
            torch:     torch 版本号,无 torch 时为 None;
            cuda:      torch 对应的 CUDA 版本(仅在有 torch 时存在);
            gpu:       首块 GPU 名,无 GPU 时为 "cpu"(仅在有 torch 时存在)。
    """
    info = {"timestamp": datetime.now(timezone.utc).isoformat()}
    try:
        import torch
        info["torch"] = torch.__version__      # 记录版本,便于日后排查差异
        info["cuda"] = torch.version.cuda      # CUDA toolkit 版本
        info["gpu"] = (                        # GPU 型号(如 A100),无卡则 cpu
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        )
    except ImportError:
        info["torch"] = None                   # 本地无 torch:仅留时间戳
    return info


def save_results(results: dict, out_dir: str, exp_id: str) -> str:
    """把实验指标 + 环境信息落成 results/<exp_id>.json。

    参数:
        results: 指标字典(如 {"ppl": {...}, "latency": {...}})。
        out_dir: 输出目录,不存在会自动创建。
        exp_id:  实验唯一标识,同时作为文件名。
    返回:
        写入的 json 文件绝对/相对路径。

    落盘内容 = {exp_id, env(环境), **results},统一结构便于后续脚本汇总成表。
    """
    os.makedirs(out_dir, exist_ok=True)              # 目录幂等创建
    # 组装 payload:exp_id 与 env 固定在前,业务指标展开在后
    payload = {"exp_id": exp_id, "env": env_info(), **results}
    path = os.path.join(out_dir, f"{exp_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        # ensure_ascii=False 保留中文;indent=2 便于人工阅读 diff
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path
