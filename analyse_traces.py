"""
analyse_traces.py
=================
Load generated trace JSONL files, segment each trace, then run
step-level informativeness + redundancy analysis.

Outputs
-------
outputs/analysis/<dataset>_<model>_analysis.jsonl  — per-example detail
outputs/analysis/<dataset>_<model>_summary.json    — aggregate stats

Usage:
    python analyse_traces.py \\
        --trace-files outputs/traces/aime24_deepseek*.jsonl \\
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \\
        --segmenter auto \\
        --output-dir outputs/analysis
"""
from __future__ import annotations
import argparse, json, logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import glob, statistics

from segmentation_ext import build_segmenter_for_model
from step_analysis import (
    StepLengthConfig, RedundancyConfig, analyse_trace,
    TraceAnalysis, score_step_length
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--trace-files", nargs="+", default=[],
                   help="Glob patterns or explicit paths to trace JSONL files.")
    p.add_argument("--trace-glob", default="",
                   help="Shell glob for trace files (alternative to --trace-files).")
    p.add_argument("--model", default="",
                   help="Model name for segmenter selection (auto-detected from file if empty).")
    p.add_argument("--segmenter", default="auto",
                   choices=["auto", "hybrid", "novelty", "sentence"],
                   help="Segmenter strategy.")
    p.add_argument("--output-dir", default="outputs/analysis")
    p.add_argument("--uninformative-token-max", type=int, default=40)
    p.add_argument("--informative-token-min",   type=int, default=120)
    p.add_argument("--jaccard-thresh",     type=float, default=0.55)
    p.add_argument("--cosine-thresh",      type=float, default=0.70)
    p.add_argument("--max-examples", type=int, default=-1,
                   help="Limit examples per file (-1 = all).")
    return p.parse_args()

def _load_traces(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]


def _trace_to_serialisable(ta: TraceAnalysis) -> Dict[str, Any]:
    """Convert TraceAnalysis to JSON-safe dict."""
    return {
        "summary": ta.summary,
        "step_lengths": [asdict(ls) for ls in ta.step_lengths],
        "per_step_redundancy": ta.per_step_redundancy,
        "informative_idx": ta.informative_idx,
        "uninformative_idx": ta.uninformative_idx,
        "redundant_pairs": [list(rp) for rp in ta.redundant_pairs],
    }


def _aggregate(all_summaries: List[Dict]) -> Dict:
    if not all_summaries:
        return {}
    keys = ["n_steps", "n_informative", "n_uninformative", "n_marginal",
            "n_redundant_pairs", "avg_redundancy_score",
            "avg_token_len", "informativeness_ratio"]
    agg = {}
    for k in keys:
        vals = [s[k] for s in all_summaries if k in s]
        if vals:
            agg[f"mean_{k}"] = round(statistics.mean(vals), 4)
            agg[f"median_{k}"] = round(statistics.median(vals), 4)
            if len(vals) > 1:
                agg[f"stdev_{k}"] = round(statistics.stdev(vals), 4)
    agg["n_examples"] = len(all_summaries)
    return agg


def analyse_file(trace_path: Path, model_name: str, segmenter_strategy: str,
                 length_cfg: StepLengthConfig, redund_cfg: RedundancyConfig,
                 output_dir: Path, max_examples: int) -> Dict:
    traces = _load_traces(trace_path)
    if max_examples > 0:
        traces = traces[:max_examples]

    # Auto-detect model from record if not supplied
    if not model_name and traces:
        model_name = traces[0].get("model", "unknown")

    segmenter = build_segmenter_for_model(model_name, strategy=segmenter_strategy)
    stem = trace_path.stem
    out_jsonl = output_dir / f"{stem}_analysis.jsonl"
    summaries: List[Dict] = []

    with out_jsonl.open("w", encoding="utf-8") as fout:
        for rec in traces:
            think = rec.get("think_text", rec.get("full_text", ""))
            steps = segmenter.split(think)
            if not steps:
                steps = [think]
            ta = analyse_trace(steps, length_cfg, redund_cfg)
            row = {
                "idx":     rec.get("idx"),
                "dataset": rec.get("dataset", stem),
                "model":   model_name,
                "n_steps": len(steps),
                "steps":   steps,
                "analysis": _trace_to_serialisable(ta),
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            summaries.append(ta.summary)
            LOGGER.info("idx=%s  n_steps=%d  informative=%d  uninformative=%d  red_pairs=%d",
                        rec.get("idx"), ta.summary["n_steps"],
                        ta.summary["n_informative"], ta.summary["n_uninformative"],
                        ta.summary["n_redundant_pairs"])

    agg = _aggregate(summaries)
    agg["source_file"] = str(trace_path)
    agg["model"] = model_name
    agg["segmenter"] = segmenter_strategy
    return agg

def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all trace files
    files: List[Path] = []
    for pattern in args.trace_files:
        files.extend(Path(p) for p in glob.glob(pattern))
    if args.trace_glob:
        files.extend(Path(p) for p in glob.glob(args.trace_glob))
    files = sorted(set(files))

    if not files:
        LOGGER.error("No trace files found. Check --trace-files or --trace-glob.")
        return

    length_cfg = StepLengthConfig(
        uninformative_token_max=args.uninformative_token_max,
        informative_token_min=args.informative_token_min,
    )
    redund_cfg = RedundancyConfig(
        jaccard_thresh=args.jaccard_thresh,
        cosine_thresh=args.cosine_thresh,
    )

    all_agg: List[Dict] = []
    for fpath in files:
        LOGGER.info("Analysing %s …", fpath)
        try:
            agg = analyse_file(fpath, model_name=args.model,
                               segmenter_strategy=args.segmenter,
                               length_cfg=length_cfg, redund_cfg=redund_cfg,
                               output_dir=output_dir,
                               max_examples=args.max_examples)
            all_agg.append(agg)
        except Exception as e:
            LOGGER.error("Failed %s: %s", fpath, e)

    summary_path = output_dir / "all_summary.json"
    summary_path.write_text(
        json.dumps({"files": all_agg}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    LOGGER.info("Summary written to %s", summary_path)
    for row in all_agg:
        LOGGER.info("  %s  n_examples=%s  mean_n_steps=%s  mean_informativeness=%s",
                    Path(row.get("source_file","?")).name,
                    row.get("n_examples"),
                    row.get("mean_n_steps"),
                    row.get("mean_informativeness_ratio"))


if __name__ == "__main__":
    main()
