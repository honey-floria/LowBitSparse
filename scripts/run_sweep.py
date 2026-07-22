"""M1 扫描脚本:自动跑 method × bits × group_size 全组合,落盘结果 json。

用法(Colab / A100 上运行):
    python scripts/run_sweep.py [--config configs/qwen0.5b_int4.yaml] \
        [--smoke]         # smoke 时每组只评 4 个窗口,快速验证流程可跑通
        [--only awq]      # 只跑指定方法(rtn/gptq/awq),如重跑 AWQ 刷新结果
        [--no-emb]        # 跳过 embedding 量化消融(EMB_GRID)

脚本根据 BASE_GRID(+ EMB_GRID)定义的超参网格,逐一构造 QuantConfig,调用
main.py 的量化+评测逻辑,结果写入 results/<exp_id>.json。
所有组合跑完后打印一行总结,告知几成功几失败,方便 Colab 监控。
"""
import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- 扫描网格定义 ---
BASE_GRID = [
    # (method,  n_bits, group_size,  symmetric)
    ("rtn",   8,  128,  False),
    ("rtn",   4,  64,   False),
    ("rtn",   4,  128,  False),
    ("rtn",   4,  256,  False),
    ("rtn",   4,  -1,   False),   # per-channel
    ("rtn",   3,  128,  False),
    ("gptq",  4,  64,   False),
    ("gptq",  4,  128,  False),
    ("gptq",  4,  256,  False),
    ("gptq",  4,  -1,   False),   # per-channel
    ("gptq",  3,  128,  False),
    ("awq",   4,  64,   False),
    ("awq",   4,  128,  False),
    ("awq",   4,  256,  False),
    ("awq",   4,  -1,   False),   # per-channel
    ("awq",   3,  128,  False),
]


# --- embedding 量化消融:linears 固定 GPTQ INT4 g128,只扫 embedding 位宽 ---
# 单一变量隔离 embedding 对压缩比/PPL 的贡献,回答"能否破 2.4x"。
EMB_GRID = [
    # (method, n_bits, group_size, symmetric, embedding_bits)
    ("gptq",  4,  128,  False,  8),   # embedding INT8(保守)
    ("gptq",  4,  128,  False,  4),   # embedding INT4(激进,目标 ~3.76x)
]


def _exp_id(method, n_bits, group_size, embedding_bits=None):
    gs = "pc" if group_size == -1 else str(group_size)
    base = f"m1_{method}_int{n_bits}_g{gs}"
    return base + (f"_embint{embedding_bits}" if embedding_bits else "")


def run_one(base_cfg, method, n_bits, group_size, sym, smoke,
            embedding_bits=None):
    """跑单个组合(可选量化 embedding),返回成功/失败。"""
    import copy
    from lowbitsparse.utils import get_logger, load_config, set_seed, save_results
    from lowbitsparse.models import load_model_and_tokenizer, model_size_report
    from lowbitsparse.quant import (
        QuantConfig, apply_quantization, compression_report,
        target_linear_names, get_calib_inputs, collect_calib_stats,
        free_calib_stats)
    import torch

    log = get_logger()
    cfg = copy.deepcopy(base_cfg)
    qcfg = QuantConfig(method=method, n_bits=n_bits, group_size=group_size,
                       symmetric=sym, skip=("lm_head",),
                       quant_embedding=embedding_bits is not None,
                       embedding_bits=embedding_bits)
    exp = _exp_id(method, n_bits, group_size, embedding_bits)
    log.info("=== 开始 %s ===", exp)

    m = cfg.get("model", {})
    model, tok = load_model_and_tokenizer(
        m.get("name", "Qwen/Qwen2.5-0.5B-Instruct"),
        dtype=m.get("dtype", "float16"), device=m.get("device", "cuda"))
    fp16 = model_size_report(model)
    set_seed(cfg.get("seed", 42))

    calib_stats = None
    if method != "rtn":
        calib_ids = get_calib_inputs(tok, n_samples=qcfg.calib_n_samples,
                                     seqlen=qcfg.calib_seqlen)
        names = target_linear_names(model, qcfg)
        calib_stats = collect_calib_stats(model, calib_ids, names)

    model, n = apply_quantization(model, qcfg, calib_stats=calib_stats)
    free_calib_stats(calib_stats)            # 评测前释放校准 Hessian,压低显存峰值
    comp = compression_report(model)
    comp["ratio"] = round(fp16["size_mb"] / comp["size_mb"], 3)

    from lowbitsparse.eval import eval_wikitext2_ppl
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    ev = cfg.get("eval", {})
    max_s = 4 if smoke else ev.get("max_samples")
    ppl = eval_wikitext2_ppl(model, tok, seqlen=ev.get("seqlen", 2048),
                              stride=ev.get("stride", 2048), max_samples=max_s)

    results = {"config": {"method": method, "n_bits": n_bits,
                          "group_size": group_size,
                          "embedding_bits": embedding_bits},
               "size_fp16": fp16, "compression": comp, "ppl": ppl}
    save_results(results, cfg.get("out_dir", "results"), exp)
    log.info("=== 完成 %s: PPL=%.4f, ratio=%.2fx ===",
             exp, ppl["ppl"], comp["ratio"])
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()             # 释放上一组显存,避免累积 OOM
    return True


def main():
    from lowbitsparse.utils import get_logger, load_config
    p = argparse.ArgumentParser(description="M1 量化扫描")
    p.add_argument("--config", default="configs/qwen0.5b_int4.yaml")
    p.add_argument("--smoke", action="store_true", help="每组只评 4 个窗口")
    p.add_argument("--only", choices=("rtn", "gptq", "awq"), default=None,
                   help="只跑指定方法的组合(如 --only awq 重跑 AWQ);默认跑全部")
    p.add_argument("--no-emb", action="store_true",
                   help="跳过 embedding 量化消融(EMB_GRID)")
    args = p.parse_args()

    log = get_logger()
    base_cfg = load_config(args.config)

    # 按 --only 过滤两张网格(EMB_GRID 元组多一个 embedding_bits 字段)
    base_grid = [g for g in BASE_GRID if args.only is None or g[0] == args.only]
    emb_grid = [] if args.no_emb else [
        g for g in EMB_GRID if args.only is None or g[0] == args.only]
    total = len(base_grid) + len(emb_grid)
    if total == 0:
        log.warning("--only %s 未匹配任何组合,退出", args.only)
        return
    log.info("本次将跑 %d 组(--only=%s, no_emb=%s)",
             total, args.only, args.no_emb)

    ok = fail = 0
    for method, n_bits, gs, sym in base_grid:
        try:
            run_one(base_cfg, method, n_bits, gs, sym, args.smoke)
            ok += 1
        except Exception:
            fail += 1
            log.error("组合 %s 失败:\n%s",
                      _exp_id(method, n_bits, gs), traceback.format_exc())
    # embedding 量化消融(linears 固定 GPTQ INT4 g128,只变 embedding 位宽)
    for method, n_bits, gs, sym, e_bits in emb_grid:
        try:
            run_one(base_cfg, method, n_bits, gs, sym, args.smoke,
                    embedding_bits=e_bits)
            ok += 1
        except Exception:
            fail += 1
            log.error("组合 %s 失败:\n%s",
                      _exp_id(method, n_bits, gs, e_bits),
                      traceback.format_exc())
    log.info("扫描结束:成功 %d, 失败 %d(共 %d 组)", ok, fail, total)


if __name__ == "__main__":
    main()
