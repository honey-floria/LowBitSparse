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


def cmd_eval(args):
    """M0 子命令:加载 FP16 模型,评测 PPL + 延迟 + 显存,并落盘为基线。

    参数:
        args: argparse 解析结果,需含 args.config(YAML 配置路径)。

    流程:读配置 → 固定种子 → 加载模型 → 统计体积 → PPL → 延迟 → 显存 → 存 json。
    """
    import torch   # 延迟导入:仅本子命令真正需要 GPU/张量

    # 同样延迟导入模型与评测模块(内部依赖 torch/transformers/datasets)
    from lowbitsparse.models import load_model_and_tokenizer, model_size_report
    from lowbitsparse.eval import (
        eval_wikitext2_ppl,   # WikiText-2 困惑度
        profile_latency,      # prefill/decode 延迟
        profile_memory,       # 显存峰值
    )

    # 读取 YAML 配置;无 --config 时用空 dict,后续 .get() 全部走默认值
    cfg = load_config(args.config) if args.config else {}
    set_seed(cfg.get("seed", 42))     # 固定随机源,保证可复现
    m = cfg.get("model", {})          # 模型相关子配置

    # 加载模型与分词器(配置缺省时回退到 0.5B / float16 / cuda)
    model, tok = load_model_and_tokenizer(
        model_name=m.get("name", "Qwen/Qwen2.5-0.5B-Instruct"),
        dtype=m.get("dtype", "float16"),
        device=m.get("device", "cuda"),
    )
    log.info("模型已加载: %s", m.get("name"))

    # 清零显存峰值统计,使随后 profile_memory 得到的是"本次评测"的峰值
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # 1) 体积:参数量与理论字节数,后续量化以此算压缩比
    size = model_size_report(model)
    log.info("参数量 %.3fM, 体积 %.1fMB", size["params_millions"], size["size_mb"])

    # 2) 精度:WikiText-2 困惑度
    ev = cfg.get("eval", {})          # 评测子配置
    ppl = eval_wikitext2_ppl(
        model, tok,
        seqlen=ev.get("seqlen", 2048),
        stride=ev.get("stride", 2048),
        max_samples=ev.get("max_samples"),   # None=全量,整数=冒烟
    )
    log.info("WikiText-2 PPL = %s", ppl["ppl"])

    # 3) 效率:prefill 延迟 + decode 吞吐
    pf = cfg.get("profile", {})       # 性能测试子配置
    lat = profile_latency(
        model, tok,
        prefill_len=pf.get("prefill_len", 512),
        decode_tokens=pf.get("decode_tokens", 128),
    )
    # 4) 显存峰值(须在上面 reset 之后、评测之后读取)
    mem = profile_memory()
    log.info("prefill %.1fms, decode %.1f tok/s, peak %sMB",
             lat["prefill_ms_median"], lat["decode_tps_median"], mem["peak_mb"])

    # 汇总四类指标 + 原始配置,统一落盘为 results/<exp_id>.json
    results = {"config": cfg, "size": size, "ppl": ppl,
               "latency": lat, "memory": mem}
    path = save_results(results, cfg.get("out_dir", "results"),
                        cfg.get("exp_id", "m0_fp16_baseline"))
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

    # quant/sparse/distill:后续里程碑,先注册占位以保证 CLI 结构完整
    for name in ("quant", "sparse", "distill"):
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
