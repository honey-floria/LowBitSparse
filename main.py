"""LowBitSparse CLI 入口。

统一命令行入口,按子命令分派到各里程碑的功能:
    python main.py eval    --config configs/qwen0.5b_base.yaml   # M0: 基线评测(已实现)
    python main.py quant   ...   # M1: 权重量化(占位)
    python main.py sparse  ...   # M2: 稀疏注意力(占位)
    python main.py distill ...   # M3: 量化感知蒸馏(占位)

设计约定:torch / 模型等重依赖在子命令函数内部延迟导入,
使本文件在无 GPU 的本地也能被导入、供 argparse 接线测试。
"""
import argparse   # 标准库参数解析,零额外依赖

# 仅从 utils 引入轻量工具(顶层不含 torch),保证本模块可在本地导入
from lowbitsparse.utils import get_logger, load_config, set_seed, save_results

log = get_logger()   # 全局 logger,各子命令共用同一实例


def _load(cfg):
    """按配置加载模型与分词器(eval/quant 共用)。"""
    from lowbitsparse.models import load_model_and_tokenizer
    m = cfg.get("model", {})
    model, tok = load_model_and_tokenizer(
        model_name=m.get("name", "Qwen/Qwen2.5-0.5B-Instruct"),
        dtype=m.get("dtype", "float16"),
        device=m.get("device", "cuda"),
    )
    log.info("模型已加载: %s", m.get("name"))
    return model, tok


def _evaluate(model, tok, cfg):
    """跑 PPL + 延迟 + 显存,返回指标 dict(eval/quant 共用的评测闭环)。"""
    import torch
    from lowbitsparse.eval import (
        eval_wikitext2_ppl, profile_latency, profile_memory)

    # 清零峰值统计,确保 profile_memory 反映"本次评测"的峰值
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    ev = cfg.get("eval", {})
    ppl = eval_wikitext2_ppl(
        model, tok,
        seqlen=ev.get("seqlen", 2048),
        stride=ev.get("stride", 2048),
        max_samples=ev.get("max_samples"),
    )
    log.info("WikiText-2 PPL = %s", ppl["ppl"])

    pf = cfg.get("profile", {})
    lat = profile_latency(
        model, tok,
        prefill_len=pf.get("prefill_len", 512),
        decode_tokens=pf.get("decode_tokens", 128),
    )
    mem = profile_memory()
    log.info("prefill %.1fms, decode %.1f tok/s, peak %sMB",
             lat["prefill_ms_median"], lat["decode_tps_median"], mem["peak_mb"])
    return {"ppl": ppl, "latency": lat, "memory": mem}


def cmd_eval(args):
    """M0 子命令:加载 FP16 模型,评测并落盘为基线。"""
    from lowbitsparse.models import model_size_report

    cfg = load_config(args.config) if args.config else {}
    set_seed(cfg.get("seed", 42))
    model, tok = _load(cfg)
    size = model_size_report(model)   # FP16 体积:压缩比分母
    log.info("参数量 %.3fM, 体积 %.1fMB", size["params_millions"], size["size_mb"])
    metrics = _evaluate(model, tok, cfg)

    results = {"config": cfg, "size": size, **metrics}
    path = save_results(results, cfg.get("out_dir", "results"),
                        cfg.get("exp_id", "m0_fp16_baseline"))
    log.info("结果已保存: %s", path)


def cmd_quant(args):
    """M1 子命令:RTN 伪量化 → 评测 → 报告压缩比,与 FP16 基线对比。"""
    from lowbitsparse.models import model_size_report
    from lowbitsparse.quant import (
        QuantConfig, apply_quantization, compression_report)

    cfg = load_config(args.config) if args.config else {}
    set_seed(cfg.get("seed", 42))
    model, tok = _load(cfg)

    fp16 = model_size_report(model)          # 量化前体积(分母)
    qcfg = QuantConfig.from_dict(cfg.get("quant", {}))
    model, n = apply_quantization(model, qcfg)   # 就地替换 Linear
    log.info("已量化 %d 个 Linear (bits=%d, group=%d, sym=%s)",
             n, qcfg.n_bits, qcfg.group_size, qcfg.symmetric)

    comp = compression_report(model)         # 量化后理论体积 + 等效 bit
    comp["ratio"] = round(fp16["size_mb"] / comp["size_mb"], 3)  # 压缩比
    log.info("量化后 %.1fMB, 等效 %.2f bit, 压缩比 %.2fx",
             comp["size_mb"], comp["effective_bits"], comp["ratio"])

    metrics = _evaluate(model, tok, cfg)     # 复用评测闭环
    results = {"config": cfg, "size_fp16": fp16, "compression": comp, **metrics}
    path = save_results(results, cfg.get("out_dir", "results"),
                        cfg.get("exp_id", "m1_rtn"))
    log.info("结果已保存: %s", path)


def _todo(name):
    """生成"尚未实现"占位处理函数的工厂。

    参数:
        name: 子命令名(quant / sparse / distill)。
    返回:
        一个接收 args 的函数;被调用即抛 SystemExit 给出友好提示,
        避免用户在里程碑未完成时静默得到错误结果。
    """
    def _fn(args):
        raise SystemExit(f"[{name}] 尚未实现,将在对应里程碑完成。")
    return _fn


def build_parser():
    """构建 argparse 解析器:一个主命令 + 四个子命令。

    返回:
        配置好的 ArgumentParser。每个子命令通过 set_defaults(func=...)
        绑定其处理函数,main() 里统一调用 args.func(args) 分派。
    """
    p = argparse.ArgumentParser(description="LowBitSparse 压缩工具箱")
    # required=True:必须给出子命令,否则报错并打印帮助
    sub = p.add_subparsers(dest="command", required=True)

    # eval:M0 已实现,默认读 0.5B 基线配置
    pe = sub.add_parser("eval", help="M0: 评测 PPL / 延迟 / 显存")
    pe.add_argument("--config", type=str, default="configs/qwen0.5b_base.yaml")
    pe.set_defaults(func=cmd_eval)

    # quant:M1 已实现,RTN 伪量化 + 压缩比评测
    pq = sub.add_parser("quant", help="M1: RTN 权重量化 + 评测")
    pq.add_argument("--config", type=str, default="configs/qwen0.5b_int4.yaml")
    pq.set_defaults(func=cmd_quant)

    # sparse/distill:后续里程碑,先注册占位以保证 CLI 结构完整
    for name in ("sparse", "distill"):
        sp = sub.add_parser(name, help=f"{name} (后续里程碑)")
        sp.add_argument("--config", type=str, default=None)
        sp.set_defaults(func=_todo(name))
    return p


def main():
    """解析命令行并分派到对应子命令处理函数。"""
    args = build_parser().parse_args()   # 解析 argv
    args.func(args)                      # 调用子命令绑定的处理函数


if __name__ == "__main__":
    # 作为脚本运行时的入口;被 import 时不执行,便于测试
    main()
