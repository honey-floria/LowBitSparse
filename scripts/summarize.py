"""M1 验收汇总:扫描 results/*.json,生成方法×bit×group 的对比表格。

用法:
    python scripts/summarize.py [--results results] [--out results/m1_summary.md]

读取每个实验 json 的压缩比 / 等效 bit / PPL,和 FP16 基线对比,
输出一张 markdown 表格(压缩比 + 精度退化),作为 M1 验收的三类结论之一。
不做作图(交给后续脚本);此处只保证"从 json 一键生成表格,避免手工作表"。
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_all(results_dir):
    """读取目录下所有 json,返回 (baseline_ppl, [行 dict])。"""
    rows = []
    baseline_ppl = None
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        exp = d.get("exp_id", os.path.basename(path)[:-5])
        ppl = (d.get("ppl") or {}).get("ppl")
        if exp == "m0_fp16_baseline" or "fp16" in exp:
            baseline_ppl = ppl
            rows.append({"exp": exp, "method": "fp16", "bits": 16,
                         "group": "-", "eff_bits": 16.0,
                         "size_mb": (d.get("size") or {}).get("size_mb"),
                         "ratio": 1.0, "ppl": ppl})
            continue
        comp = d.get("compression", {})
        cfg = d.get("config", {})
        # 兼容两种落盘结构:run_sweep 存扁平 config,cmd_quant 存完整 yaml(含 quant 段)
        qc = cfg.get("quant", cfg)
        rows.append({
            "exp": exp,
            "method": qc.get("method", "?"),
            "bits": qc.get("n_bits", "?"),
            "group": qc.get("group_size", "?"),
            "eff_bits": comp.get("effective_bits"),
            "size_mb": comp.get("size_mb"),
            "ratio": comp.get("ratio"),
            "ppl": ppl,
        })
    return baseline_ppl, rows


def _fmt(v, nd=3):
    """None 显示为 '-',数值保留 nd 位小数。"""
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def build_table(baseline_ppl, rows):
    """把行数据渲染成 markdown 表格字符串(含相对基线的 ΔPPL)。"""
    head = ("| 实验 | 方法 | bit | group | 等效bit | 体积MB | 压缩比 | PPL | ΔPPL |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
    # 排序:fp16 在前,其余按 方法→bit→group
    def key(r):
        return (0 if r["method"] == "fp16" else 1, r["method"],
                -(r["bits"] if isinstance(r["bits"], int) else 0), str(r["group"]))
    lines = []
    for r in sorted(rows, key=key):
        dppl = "-"
        if baseline_ppl is not None and r["ppl"] is not None and r["method"] != "fp16":
            dppl = f"{r['ppl'] - baseline_ppl:+.3f}"
        lines.append(
            f"| {r['exp']} | {r['method']} | {r['bits']} | {r['group']} | "
            f"{_fmt(r['eff_bits'])} | {_fmt(r['size_mb'],1)} | "
            f"{_fmt(r['ratio'],2)}x | {_fmt(r['ppl'],4)} | {dppl} |")
    return head + "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description="M1 结果汇总为表格")
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="results/m1_summary.md")
    args = p.parse_args()

    baseline_ppl, rows = _load_all(args.results)
    table = build_table(baseline_ppl, rows)
    header = "# M1 权重量化验收汇总\n\n"
    if baseline_ppl is not None:
        header += f"FP16 基线 PPL = {baseline_ppl}\n\n"
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(header + table)
    print(header + table)
    print(f"[已写入] {args.out}")


if __name__ == "__main__":
    main()
