"""LowBitSparse 命令行入口。

已实现命令:
    python main.py eval   --config configs/qwen0.5b_base.yaml
    python main.py quant  --config configs/qwen0.5b_int4.yaml
    python main.py sparse --config configs/qwen0.5b_sparse_sliding.yaml

torch、transformers、datasets 等重依赖延迟到子命令内部导入,
保证本文件在 CPU-only 环境和轻量测试中仍可导入。
"""
import argparse

from lowbitsparse.utils import get_logger, load_config, set_seed, save_results

log = get_logger()


def _load(cfg):
    """加载 eval / quant 共用的模型和分词器。"""
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
    """执行通用评测流程:WikiText-2 PPL、延迟、显存。"""
    import torch
    from lowbitsparse.eval import (
        eval_wikitext2_ppl, profile_latency, profile_memory)

    # 只统计本次评测阶段的峰值显存,不混入模型加载峰值。
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


def _run_sparse(model, tok, cfg):
    """M2:稀疏注意力长序列基准。"""
    from lowbitsparse.sparse import SparseConfig
    from lowbitsparse.sparse.benchmark import benchmark_sparse_attention

    scfg = SparseConfig.from_dict(cfg.get("sparse", {}))
    log.info("稀疏注意力配置: mode=%s, window=%d, sink=%d, block=%d, cache_pruning=%s",
             scfg.mode, scfg.window_size, scfg.sink_size, scfg.block_size, scfg.cache_pruning)
    return benchmark_sparse_attention(
        model, tok, scfg,
        eval_cfg=cfg.get("eval", {}),
        profile_cfg=cfg.get("profile", {}),
    )


def cmd_eval(args):
    """M0:评测 FP16 基线并落盘指标。"""
    from lowbitsparse.models import model_size_report

    cfg = load_config(args.config) if args.config else {}
    set_seed(cfg.get("seed", 42))
    model, tok = _load(cfg)
    size = model_size_report(model)
    log.info("参数量 %.3fM, 体积 %.1fMB", size["params_millions"], size["size_mb"])
    metrics = _evaluate(model, tok, cfg)

    results = {"config": cfg, "size": size, **metrics}
    path = save_results(results, cfg.get("out_dir", "results"),
                        cfg.get("exp_id", "m0_fp16_baseline"))
    log.info("结果已保存: %s", path)


def _run_calibration(model, tok, qcfg, cfg):
    """为 GPTQ/AWQ 收集逐层校准统计;RTN 不需要校准。"""
    if qcfg.method == "rtn":
        return None
    from lowbitsparse.quant import (
        get_calib_inputs, collect_calib_stats, target_linear_names)
    log.info("收集校准统计 (method=%s, n=%d, seqlen=%d)...",
             qcfg.method, qcfg.calib_n_samples, qcfg.calib_seqlen)
    calib_ids = get_calib_inputs(tok, n_samples=qcfg.calib_n_samples,
                                 seqlen=qcfg.calib_seqlen)
    names = target_linear_names(model, qcfg)
    stats = collect_calib_stats(model, calib_ids, names)
    log.info("校准完成,共 %d 层有统计", len(stats))
    return stats


def cmd_quant(args):
    """M1:执行权重量化、精度评测和压缩比统计。"""
    from lowbitsparse.models import model_size_report
    from lowbitsparse.quant import (
        QuantConfig, apply_quantization, compression_report, free_calib_stats)

    cfg = load_config(args.config) if args.config else {}
    set_seed(cfg.get("seed", 42))
    model, tok = _load(cfg)

    fp16 = model_size_report(model)
    qcfg = QuantConfig.from_dict(cfg.get("quant", {}))
    calib_stats = _run_calibration(model, tok, qcfg, cfg)
    model, n = apply_quantization(model, qcfg, calib_stats=calib_stats)
    log.info("已量化 %d 个 Linear (method=%s, bits=%d, group=%d, sym=%s)",
             n, qcfg.method, qcfg.n_bits, qcfg.group_size, qcfg.symmetric)

    # 校准统计已用完,评测前释放,避免 Hessian 常驻拉高显存峰值。
    freed = free_calib_stats(calib_stats)
    if freed:
        log.info("已释放校准统计 %.1f MB", freed)

    comp = compression_report(model)
    comp["ratio"] = round(fp16["size_mb"] / comp["size_mb"], 3)
    log.info("量化后 %.1fMB, 等效 %.2f bit, 压缩比 %.2fx",
             comp["size_mb"], comp["effective_bits"], comp["ratio"])

    metrics = _evaluate(model, tok, cfg)
    results = {"config": cfg, "size_fp16": fp16, "compression": comp, **metrics}
    path = save_results(results, cfg.get("out_dir", "results"),
                        cfg.get("exp_id", f"m1_{qcfg.method}"))
    log.info("结果已保存: %s", path)


def cmd_sparse(args):
    """M2:稀疏注意力 baseline vs sparse 长序列基准。"""
    from lowbitsparse.models import model_size_report

    cfg = load_config(args.config) if args.config else {}
    set_seed(cfg.get("seed", 42))
    model, tok = _load(cfg)
    size = model_size_report(model)
    log.info("参数量 %.3fM, 体积 %.1fMB", size["params_millions"], size["size_mb"])
    metrics = _run_sparse(model, tok, cfg)

    results = {"config": cfg, "size": size, **metrics}
    path = save_results(results, cfg.get("out_dir", "results"),
                        cfg.get("exp_id", "m2_sparse_sliding"))
    log.info("结果已保存: %s", path)


def _todo(name):
    """为已注册但尚未实现的里程碑命令构造占位处理函数。"""
    def _fn(args):
        raise SystemExit(f"[{name}] 尚未实现,将在对应里程碑完成。")
    return _fn


def build_parser():
    """构建顶层解析器,并把子命令绑定到对应处理函数。"""
    p = argparse.ArgumentParser(description="LowBitSparse 压缩工具箱")
    sub = p.add_subparsers(dest="command", required=True)

    pe = sub.add_parser("eval", help="M0: 评测 PPL / 延迟 / 显存")
    pe.add_argument("--config", type=str, default="configs/qwen0.5b_base.yaml")
    pe.set_defaults(func=cmd_eval)

    pq = sub.add_parser("quant", help="M1: 权重量化 + 评测")
    pq.add_argument("--config", type=str, default="configs/qwen0.5b_int4.yaml")
    pq.set_defaults(func=cmd_quant)

    ps = sub.add_parser("sparse", help="M2: 稀疏注意力长序列基准")
    ps.add_argument("--config", type=str, default="configs/qwen0.5b_sparse_sliding.yaml")
    ps.set_defaults(func=cmd_sparse)

    sp = sub.add_parser("distill", help="M3: 量化感知蒸馏 (后续里程碑)")
    sp.add_argument("--config", type=str, default=None)
    sp.set_defaults(func=_todo("distill"))
    return p


def main():
    """解析命令行参数并分派到选中的子命令。"""
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
