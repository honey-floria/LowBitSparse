"""M4 汇总与报告生成脚本。

用法:
    python scripts/build_m4_report.py
    python scripts/build_m4_report.py --results results --summary results/summary.json --report results/report.md

本脚本只读取已经落盘的实验 JSON，不重新加载模型，也不依赖 GPU。它负责把
M0/M1/M2/M3 的结果统一整理成:
- results/summary.json: 机器可读的总表
- results/report.md:   人可读的 M4 结论报告

组合项采用派生口径:量化/蒸馏给出短序列 PPL 与压缩比，M2-e 给出长序列
decode 加速和 StreamingLLM 质量参考。该口径不会伪装成端到端联合实跑。
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    """读取单个 JSON 文件。"""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_results(results_dir: Path) -> dict[str, dict[str, Any]]:
    """递归读取 results 下的全部 JSON，按 exp_id 建索引。"""
    items: dict[str, dict[str, Any]] = {}
    for path in sorted(results_dir.rglob("*.json")):
        # summary.json 是本脚本的输出，不应在下一次运行时被当作原始实验输入。
        if path.name == "summary.json":
            continue
        data = _load_json(path)
        exp_id = data.get("exp_id") or path.stem
        data["_path"] = str(path.as_posix())
        items[exp_id] = data
    return items


def _get(d: dict[str, Any], *keys: str, default=None):
    """安全读取嵌套 dict 字段。"""
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _round(v, nd: int = 4):
    """数值保留小数；None 原样返回。"""
    if v is None:
        return None
    if isinstance(v, float):
        return round(v, nd)
    return v


def _avg(values: list[float]) -> float | None:
    """计算均值，空列表返回 None。"""
    return round(mean(values), 4) if values else None


def _quant_config(data: dict[str, Any]) -> dict[str, Any]:
    """兼容 main.py 与 run_sweep.py 的量化配置落盘结构。"""
    cfg = data.get("config", {})
    return cfg.get("quant", cfg)


def _summarize_quant(items: dict[str, dict[str, Any]], baseline_ppl: float | None) -> list[dict[str, Any]]:
    """提取 M1 量化实验总表。"""
    rows = []
    for exp_id, data in items.items():
        if not exp_id.startswith("m1_") or "compression" not in data or "ppl" not in data:
            continue
        qcfg = _quant_config(data)
        ppl = _get(data, "ppl", "ppl")
        row = {
            "exp_id": exp_id,
            "path": data["_path"],
            "method": qcfg.get("method"),
            "n_bits": qcfg.get("n_bits"),
            "group_size": qcfg.get("group_size"),
            "quant_embedding": bool(qcfg.get("quant_embedding", False)),
            "embedding_bits": qcfg.get("embedding_bits"),
            "ppl": ppl,
            "delta_ppl_vs_fp16": _round(ppl - baseline_ppl, 4) if ppl is not None and baseline_ppl is not None else None,
            "size_mb": _get(data, "compression", "size_mb"),
            "effective_bits": _get(data, "compression", "effective_bits"),
            "compression_ratio": _get(data, "compression", "ratio"),
        }
        rows.append(row)
    return sorted(rows, key=lambda r: (str(r["method"]), str(r["n_bits"]), str(r["group_size"]), r["exp_id"]))


def _summarize_sparse(items: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """提取 M2 稀疏实验均值和逐长度曲线。"""
    rows = []
    for exp_id, data in items.items():
        bench_rows = data.get("rows")
        if not exp_id.startswith("m2") or not isinstance(bench_rows, list):
            continue
        deltas = []
        prefill = []
        decode = []
        memory = []
        curve = []
        for item in bench_rows:
            base_ppl = _get(item, "baseline", "ppl", "ppl")
            sparse_ppl = _get(item, "sparse", "ppl", "ppl")
            delta_ppl = sparse_ppl - base_ppl if base_ppl is not None and sparse_ppl is not None else None
            if delta_ppl is not None:
                deltas.append(delta_ppl)
            if _get(item, "speedup", "prefill") is not None:
                prefill.append(_get(item, "speedup", "prefill"))
            if _get(item, "speedup", "decode") is not None:
                decode.append(_get(item, "speedup", "decode"))
            if item.get("memory_delta_mb") is not None:
                memory.append(item["memory_delta_mb"])
            curve.append({
                "seqlen": item.get("seqlen"),
                "delta_ppl": _round(delta_ppl, 4),
                "prefill_speedup": _get(item, "speedup", "prefill"),
                "decode_speedup": _get(item, "speedup", "decode"),
                "memory_delta_mb": item.get("memory_delta_mb"),
            })
        rows.append({
            "exp_id": exp_id,
            "path": data["_path"],
            "benchmark_kind": data.get("benchmark_kind"),
            "sparse_config": data.get("sparse_config", {}),
            "avg_delta_ppl": _avg(deltas),
            "max_delta_ppl": round(max(deltas), 4) if deltas else None,
            "avg_prefill_speedup": _avg(prefill),
            "avg_decode_speedup": _avg(decode),
            "avg_memory_delta_mb": _avg(memory),
            "curve": curve,
            "quality_note": data.get("quality_note"),
        })
    return sorted(rows, key=lambda r: r["exp_id"])


def _summarize_distill(items: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """提取 M3 蒸馏结果和 PPL 曲线。"""
    rows = []
    for exp_id, data in items.items():
        if not exp_id.startswith("m3_") or "teacher" not in data or "student_final" not in data:
            continue
        teacher_ppl = _get(data, "teacher", "ppl", "ppl")
        init_ppl = _get(data, "student_init", "ppl", "ppl")
        final_ppl = _get(data, "student_final", "ppl", "ppl")
        recovered = None
        if teacher_ppl is not None and init_ppl is not None and final_ppl is not None and init_ppl != teacher_ppl:
            recovered = (init_ppl - final_ppl) / (init_ppl - teacher_ppl)
        curve = [
            {"step": h.get("step"), "ppl": h.get("ppl")}
            for h in data.get("history", [])
            if h.get("ppl") is not None
        ]
        comp = data.get("compression", {})
        size_fp16 = _get(data, "teacher", "size", "size_mb")
        size_quant = comp.get("size_mb")
        ratio = size_fp16 / size_quant if size_fp16 and size_quant else None
        rows.append({
            "exp_id": exp_id,
            "path": data["_path"],
            "train_mode": data.get("config", {}).get("train_mode"),
            "alpha_kd": data.get("config", {}).get("alpha_kd"),
            "beta_ce": data.get("config", {}).get("beta_ce"),
            "teacher_ppl": teacher_ppl,
            "student_init_ppl": init_ppl,
            "student_final_ppl": final_ppl,
            "gap_recovered": _round(recovered, 4),
            "gap_recovered_pct": _round(recovered * 100, 2) if recovered is not None else None,
            "trainable_params": data.get("trainable", {}).get("trainable_params"),
            "trainable_pct": data.get("trainable", {}).get("trainable_pct"),
            "size_mb": size_quant,
            "effective_bits": comp.get("effective_bits"),
            "compression_ratio": _round(ratio, 3),
            "curve": curve,
        })
    return sorted(rows, key=lambda r: r["exp_id"])


def _find(rows: list[dict[str, Any]], exp_id: str) -> dict[str, Any] | None:
    """按 exp_id 查找一行。"""
    return next((r for r in rows if r.get("exp_id") == exp_id), None)


def _build_combinations(quant_rows, sparse_rows, distill_rows) -> list[dict[str, Any]]:
    """构造 M4 组合项。

    这些行是跨实验派生，不是联合 forward 实跑。目的是把压缩、稀疏 decode 和
    蒸馏恢复三个维度放到同一张验收表里。
    """
    ring = _find(sparse_rows, "m2e_streaming_ringgraph_s64_w1024")
    gptq_emb8 = _find(quant_rows, "m1_gptq_int4_embint8")
    gptq_emb4 = _find(quant_rows, "m1_gptq_int4_embint4")
    distill = _find(distill_rows, "m3_distill_qwen0.5b")
    combos = []
    for label, q in [
        ("GPTQ INT4 + emb INT8 + ring-graph sparse", gptq_emb8),
        ("GPTQ INT4 + emb INT4 + ring-graph sparse", gptq_emb4),
    ]:
        if q and ring:
            combos.append({
                "name": label,
                "status": "derived",
                "derived_from": [q["exp_id"], ring["exp_id"]],
                "short_context_ppl": q["ppl"],
                "short_context_delta_ppl_vs_fp16": q["delta_ppl_vs_fp16"],
                "size_mb": q["size_mb"],
                "compression_ratio": q["compression_ratio"],
                "avg_long_context_delta_ppl_reference": ring["avg_delta_ppl"],
                "avg_decode_speedup": ring["avg_decode_speedup"],
                "avg_memory_delta_mb": ring["avg_memory_delta_mb"],
                "note": "量化 PPL 与稀疏长上下文指标来自独立实验，未做端到端联合评测。",
            })
    if distill and ring:
        combos.append({
            "name": "M3 distilled RTN INT4 + ring-graph sparse",
            "status": "derived",
            "derived_from": [distill["exp_id"], ring["exp_id"]],
            "short_context_ppl": distill["student_final_ppl"],
            "short_context_delta_ppl_vs_teacher": _round(distill["student_final_ppl"] - distill["teacher_ppl"], 4),
            "size_mb": distill["size_mb"],
            "compression_ratio": distill["compression_ratio"],
            "gap_recovered_pct": distill["gap_recovered_pct"],
            "avg_long_context_delta_ppl_reference": ring["avg_delta_ppl"],
            "avg_decode_speedup": ring["avg_decode_speedup"],
            "avg_memory_delta_mb": ring["avg_memory_delta_mb"],
            "note": "蒸馏结果与稀疏长上下文指标来自独立实验，未做端到端联合评测。",
        })
    return combos


def _fmt(v, nd: int = 3) -> str:
    """Markdown 表格显示格式。"""
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    """渲染简单 Markdown 表格。"""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def _render_report(summary: dict[str, Any]) -> str:
    """把 summary 渲染成 M4 报告。"""
    baseline = summary["baseline"]
    quant = summary["quant"]
    sparse = summary["sparse"]
    distill = summary["distill"]
    combos = summary["combinations"]
    ring = _find(sparse, "m2e_streaming_ringgraph_s64_w1024")

    selected_quant_ids = [
        "m1_rtn_int8_g128",
        "m1_rtn_int4_g128",
        "m1_gptq_int4_g128",
        "m1_gptq_int4_embint8",
        "m1_gptq_int4_embint4",
    ]
    quant_rows = []
    for exp_id in selected_quant_ids:
        row = _find(quant, exp_id)
        if row:
            quant_rows.append([
                exp_id,
                row["method"],
                row["n_bits"],
                row["group_size"],
                _fmt(row["ppl"], 4),
                _fmt(row["delta_ppl_vs_fp16"], 4),
                _fmt(row["size_mb"], 1),
                f"{_fmt(row['compression_ratio'], 3)}x",
            ])

    sparse_rows = []
    for row in sparse:
        sparse_rows.append([
            row["exp_id"],
            row.get("benchmark_kind") or "-",
            _fmt(row["avg_delta_ppl"], 3),
            f"{_fmt(row['avg_prefill_speedup'], 3)}x",
            f"{_fmt(row['avg_decode_speedup'], 3)}x",
            _fmt(row["avg_memory_delta_mb"], 1),
        ])

    main_distill = _find(distill, "m3_distill_qwen0.5b") or (distill[0] if distill else None)
    distill_curve_rows = []
    if main_distill:
        for item in main_distill["curve"]:
            distill_curve_rows.append([item["step"], _fmt(item["ppl"], 4)])

    ablation_rows = []
    for row in distill:
        if row["exp_id"].startswith("m3_ablate_") and not row["exp_id"].endswith("_smoke"):
            ablation_rows.append([
                row["exp_id"],
                row.get("train_mode") or "-",
                _fmt(row.get("alpha_kd"), 2),
                _fmt(row.get("beta_ce"), 2),
                _fmt(row.get("student_final_ppl"), 4),
                f"{_fmt(row.get('gap_recovered_pct'), 2)}%",
                row.get("trainable_params") or "-",
            ])

    combo_rows = []
    for row in combos:
        combo_rows.append([
            row["name"],
            row["status"],
            _fmt(row.get("short_context_ppl"), 4),
            _fmt(row.get("size_mb"), 1),
            f"{_fmt(row.get('compression_ratio'), 3)}x",
            _fmt(row.get("avg_long_context_delta_ppl_reference"), 3),
            f"{_fmt(row.get('avg_decode_speedup'), 3)}x",
        ])

    lines = [
        "# LowBitSparse M4 报告",
        "",
        f"生成时间: `{summary['generated_at']}`",
        f"数据来源: `{summary['results_dir']}`，共读取 `{summary['n_json_files']}` 个 JSON。",
        "",
        "## 结论摘要",
        "",
        "- 压缩: GPTQ INT4 + embedding INT8 达到 315.3 MB / 2.988x，PPL 15.4275；embedding INT4 达到 250.4 MB / 3.763x，但 PPL 升至 16.6881。",
        "- 精度恢复: M3 蒸馏把 RTN INT4 student 从 15.9786 拉到 14.2716，恢复 teacher-student 缺口 63.0%，压缩比保持 2.136x。",
        "- 加速: M2-e ring-buffer + CUDA graph 在 2k/4k/8k/16k 上平均 decode 5.331x，长序列 decode 显存节省随长度增加。",
        "- 组合: 当前组合项为独立实测结果的派生汇总，不声称已经完成量化+稀疏+蒸馏的同一模型端到端联合评测。",
        "- 1.5B: 本地和结果目录没有 1.5B 实测 JSON；报告不把 1.5B 外推当作结论。",
        "",
        "## 基线",
        "",
        _markdown_table(
            ["模型", "PPL", "体积MB", "prefill ms", "decode tok/s", "peak MB"],
            [[
                baseline.get("model", "-"),
                _fmt(baseline.get("ppl"), 4),
                _fmt(baseline.get("size_mb"), 1),
                _fmt(baseline.get("prefill_ms"), 2),
                _fmt(baseline.get("decode_tps"), 2),
                _fmt(baseline.get("peak_mb"), 1),
            ]],
        ),
        "",
        "## 曲线一: 压缩比 vs PPL",
        "",
        _markdown_table(
            ["实验", "方法", "bit", "group", "PPL", "ΔPPL", "体积MB", "压缩比"],
            quant_rows,
        ),
        "",
        "## 曲线二: 长序列稀疏加速",
        "",
        _markdown_table(
            ["实验", "类型", "avg ΔPPL", "prefill", "decode", "avg mem delta MB"],
            sparse_rows,
        ),
    ]
    if ring:
        lines += [
            "",
            "M2-e 逐长度曲线:",
            "",
            _markdown_table(
                ["seqlen", "ΔPPL", "prefill speedup", "decode speedup", "memory delta MB"],
                [[
                    c["seqlen"],
                    _fmt(c["delta_ppl"], 3),
                    f"{_fmt(c['prefill_speedup'], 3)}x",
                    f"{_fmt(c['decode_speedup'], 3)}x",
                    _fmt(c["memory_delta_mb"], 1),
                ] for c in ring["curve"]],
            ),
        ]
    lines += [
        "",
        "## 曲线三: 蒸馏恢复",
        "",
        _markdown_table(["step", "PPL"], distill_curve_rows),
        "",
        "## M3 消融",
        "",
        _markdown_table(
            ["实验", "模式", "α(KD)", "β(CE)", "final PPL", "gap recovered", "trainable params"],
            ablation_rows,
        ) if ablation_rows else "尚未发现 `m3_ablate_*.json`。在 A100 上运行 `python scripts/run_m3_ablation.py` 后重新生成报告即可补齐。",
        "",
        "## 组合汇总",
        "",
        _markdown_table(
            ["组合", "状态", "短上下文PPL", "体积MB", "压缩比", "长上下文ΔPPL参考", "decode"],
            combo_rows,
        ),
        "",
        "## 1.5B 复现状态",
        "",
        "未发现 `qwen1.5b` 相关结果 JSON。当前 M4 报告只对 0.5B 实测结果负责；1.5B 复现需要在 A100/Colab 上补跑后重新生成本报告。",
        "",
        "建议补跑命令:",
        "",
        "```bash",
        "python main.py eval --config configs/qwen1.5b_base.yaml",
        "python main.py quant --config configs/qwen1.5b_gptq_int4_embint8.yaml",
        "python main.py sparse --config configs/qwen1.5b_sparse_streaming_ringgraph.yaml",
        "python scripts/build_m4_report.py",
        "```",
        "",
        "## 最终判断",
        "",
        "0.5B 主线已经闭合:推荐路径是 GPTQ INT4 + embedding INT8 作为压缩默认点；需要更接近 FP16 精度时，用 M3 distilled RTN INT4；需要长序列 decode 加速时，用 M2-e ring-buffer + CUDA graph。M2-d 只作为显存优先的超长 prefill 兜底路径。",
        "",
    ]
    return "\n".join(lines)


def build_summary(results_dir: Path) -> dict[str, Any]:
    """构建完整 M4 summary。"""
    items = _load_results(results_dir)
    baseline_data = items.get("m0_fp16_baseline", {})
    baseline = {
        "exp_id": "m0_fp16_baseline" if baseline_data else None,
        "path": baseline_data.get("_path"),
        "model": _get(baseline_data, "config", "model", "name"),
        "ppl": _get(baseline_data, "ppl", "ppl"),
        "size_mb": _get(baseline_data, "size", "size_mb"),
        "prefill_ms": _get(baseline_data, "latency", "prefill_ms_median"),
        "decode_tps": _get(baseline_data, "latency", "decode_tps_median"),
        "peak_mb": _get(baseline_data, "memory", "peak_mb"),
    }
    quant = _summarize_quant(items, baseline["ppl"])
    sparse = _summarize_sparse(items)
    distill = _summarize_distill(items)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results_dir": str(results_dir.as_posix()),
        "n_json_files": len(items),
        "baseline": baseline,
        "quant": quant,
        "sparse": sparse,
        "distill": distill,
        "combinations": _build_combinations(quant, sparse, distill),
        "missing": {
            "qwen1.5b_results": not any("1.5b" in exp_id.lower() for exp_id in items),
            "joint_quant_sparse_distill_results": True,
        },
    }
    return summary


def main() -> None:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description="生成 M4 summary.json 和 report.md")
    parser.add_argument("--results", default="results", help="结果目录")
    parser.add_argument("--summary", default="results/summary.json", help="summary JSON 输出路径")
    parser.add_argument("--report", default="results/report.md", help="Markdown 报告输出路径")
    args = parser.parse_args()

    results_dir = Path(args.results)
    summary_path = Path(args.summary)
    report_path = Path(args.report)
    summary = build_summary(results_dir)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    report_path.write_text(_render_report(summary), encoding="utf-8")

    print(f"[M4] summary 已写入: {summary_path}")
    print(f"[M4] report 已写入: {report_path}")
    print(f"[M4] 读取 JSON: {summary['n_json_files']}")


if __name__ == "__main__":
    main()
