"""M3 蒸馏消融脚本:训练形态 × α/β 权重。

用法:
    python scripts/run_m3_ablation.py
    python scripts/run_m3_ablation.py --smoke
    python scripts/run_m3_ablation.py --modes full scale lora --loss-grid 0.7:0.3,1.0:0.0,0.5:0.5

默认网格:
- train_mode: full / scale / lora
- loss: α/β = 0.7/0.3, 1.0/0.0, 0.5/0.5

每组都会调用正式 M3 入口 `run_distillation_from_config`,独立生成
`results/m3_ablate_<mode>_aXX_bYY.json`。最后再汇总为:
- results/m3_ablation_summary.json
- results/m3_ablation_summary.md
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _tag_float(v: float) -> str:
    """把 0.7 变成 07,把 1.0 变成 10,用于 exp_id。"""
    return f"{v:.2f}".rstrip("0").rstrip(".").replace(".", "")


def _parse_loss_grid(spec: str) -> list[tuple[float, float]]:
    """解析 `0.7:0.3,1.0:0.0` 形式的 α/β 网格。"""
    pairs = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        alpha, beta = item.split(":", 1)
        pairs.append((float(alpha), float(beta)))
    if not pairs:
        raise ValueError("loss-grid 为空")
    return pairs


def _load_yaml(path: Path) -> dict[str, Any]:
    """读取 YAML 配置。"""
    from lowbitsparse.utils import load_config

    return load_config(str(path))


def _make_cfg(base: dict[str, Any], mode: str, alpha: float, beta: float, smoke: bool) -> dict[str, Any]:
    """基于正式 M3 配置生成单组消融配置。"""
    cfg = copy.deepcopy(base)
    cfg["exp_id"] = f"m3_ablate_{mode}_a{_tag_float(alpha)}_b{_tag_float(beta)}"
    distill = cfg.setdefault("distill", {})
    distill["train_mode"] = mode
    distill["alpha_kd"] = alpha
    distill["beta_ce"] = beta
    distill.setdefault("lora_rank", 8)
    distill.setdefault("lora_alpha", 16.0)
    if smoke:
        cfg["exp_id"] += "_smoke"
        distill["train_samples"] = min(int(distill.get("train_samples", 32)), 16)
        distill["eval_samples"] = min(int(distill.get("eval_samples", 8)), 4)
        distill["max_steps"] = min(int(distill.get("max_steps", 10)), 4)
        distill["eval_every"] = min(int(distill.get("eval_every", 2)), 2)
        distill["log_every"] = 1
    return cfg


def _ppl(d: dict[str, Any], *path: str) -> float | None:
    """安全读取嵌套 PPL。"""
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur if isinstance(cur, (int, float)) else None


def _summarize_one(path: Path) -> dict[str, Any]:
    """把单组蒸馏 JSON 压成消融表格行。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    cfg = data.get("config", {})
    teacher = _ppl(data, "teacher", "ppl", "ppl")
    init = _ppl(data, "student_init", "ppl", "ppl")
    final = _ppl(data, "student_final", "ppl", "ppl")
    recovered = None
    if teacher is not None and init is not None and final is not None and init != teacher:
        recovered = (init - final) / (init - teacher)
    trainable = data.get("trainable", {})
    comp = data.get("compression", {})
    size_fp16 = data.get("teacher", {}).get("size", {}).get("size_mb")
    size_quant = comp.get("size_mb")
    ratio = size_fp16 / size_quant if size_fp16 and size_quant else None
    return {
        "exp_id": data.get("exp_id", path.stem),
        "path": str(path.as_posix()),
        "train_mode": cfg.get("train_mode"),
        "alpha_kd": cfg.get("alpha_kd"),
        "beta_ce": cfg.get("beta_ce"),
        "teacher_ppl": teacher,
        "student_init_ppl": init,
        "student_final_ppl": final,
        "gap_recovered": round(recovered, 4) if recovered is not None else None,
        "gap_recovered_pct": round(recovered * 100, 2) if recovered is not None else None,
        "trainable_params": trainable.get("trainable_params"),
        "trainable_pct": trainable.get("trainable_pct"),
        "size_mb": size_quant,
        "compression_ratio": round(ratio, 3) if ratio is not None else None,
    }


def _fmt(v, nd: int = 4) -> str:
    """Markdown 显示格式。"""
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def write_summary(rows: list[dict[str, Any]], out_json: Path, out_md: Path) -> None:
    """写入 JSON 和 Markdown 消融汇总。"""
    rows = sorted(rows, key=lambda r: (str(r["train_mode"]), r["alpha_kd"], r["beta_ce"]))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# M3 蒸馏消融汇总",
        "",
        "| 实验 | 模式 | α(KD) | β(CE) | final PPL | gap recovered | trainable params | trainable % | 压缩比 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['exp_id']} | {row['train_mode']} | {_fmt(row['alpha_kd'], 2)} | "
            f"{_fmt(row['beta_ce'], 2)} | {_fmt(row['student_final_ppl'], 4)} | "
            f"{_fmt(row['gap_recovered_pct'], 2)}% | {row.get('trainable_params', '-')} | "
            f"{_fmt(row.get('trainable_pct'), 4)}% | {_fmt(row.get('compression_ratio'), 3)}x |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description="运行 M3 train_mode × alpha/beta 消融")
    parser.add_argument("--config", default="configs/qwen0.5b_distill.yaml", help="基础 M3 配置")
    parser.add_argument("--modes", nargs="+", default=["full", "scale", "lora"],
                        choices=["full", "scale", "lora"], help="训练形态网格")
    parser.add_argument("--loss-grid", default="0.7:0.3,1.0:0.0,0.5:0.5",
                        help="α/β 网格,格式如 0.7:0.3,1.0:0.0")
    parser.add_argument("--smoke", action="store_true", help="每组只跑少量 step/窗口")
    parser.add_argument("--skip-existing", action="store_true", help="已有结果 JSON 时跳过训练")
    parser.add_argument("--summary-json", default="results/m3_ablation_summary.json")
    parser.add_argument("--summary-md", default="results/m3_ablation_summary.md")
    args = parser.parse_args()

    base = _load_yaml(Path(args.config))
    loss_grid = _parse_loss_grid(args.loss_grid)
    from lowbitsparse.distill import run_distillation_from_config

    rows = []
    failures = []
    out_dir = Path(base.get("out_dir", "results"))
    for mode in args.modes:
        for alpha, beta in loss_grid:
            cfg = _make_cfg(base, mode, alpha, beta, args.smoke)
            path = out_dir / f"{cfg['exp_id']}.json"
            try:
                if args.skip_existing and path.exists():
                    print(f"[M3-ablation] skip existing: {path}")
                else:
                    print(f"[M3-ablation] run: {cfg['exp_id']}")
                    run_distillation_from_config(cfg)
                rows.append(_summarize_one(path))
            except Exception:
                failures.append({"exp_id": cfg["exp_id"], "traceback": traceback.format_exc()})
                print(f"[M3-ablation] failed: {cfg['exp_id']}")
                print(failures[-1]["traceback"])
            finally:
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

    write_summary(rows, Path(args.summary_json), Path(args.summary_md))
    if failures:
        fail_path = Path(args.summary_json).with_name("m3_ablation_failures.json")
        fail_path.write_text(json.dumps({"failures": failures}, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        print(f"[M3-ablation] failures: {len(failures)}, see {fail_path}")
    print(f"[M3-ablation] summary: {args.summary_json}")
    print(f"[M3-ablation] markdown: {args.summary_md}")


if __name__ == "__main__":
    main()
