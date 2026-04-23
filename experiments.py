from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from diagnostics import CognitiveFlowAnalyzer, GraphConstructor
from mechanistic import LatentMonitor
from mitigation import ConfusionContrastiveLogitsProcessor, FirstStepFilter, PruningEngine
from models import ReasoningModel, ReasoningModelConfig
from pipeline import ReasoningPipeline
from segmentation import build_segmenter

LOGGER = logging.getLogger(__name__)


@dataclass
class ExperimentRun:
    prompt_id: int
    run_mode: str
    segmenter: str
    unit_count: int
    repeated_unit_ratio: float
    adjacent_overlap_ratio: float
    loop_count: int
    cognitive_self_transition_rate: float
    dag_node_count: int
    dag_pruned_node_count: int
    prune_ratio: float
    latent_classification: str
    first_step_score: float
    loop_detected: int
    judge_insert_count: int
    judge_merge_count: int
    judge_total_count: int
    units: List[str]
    metadata: Dict[str, Any]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def build_default_prompt() -> str:
    return (
        "Solve this math word problem and think step by step.\n"
        "Use <think>...</think> tags for your reasoning and provide a final answer.\n\n"
        "A bakery sold 48 muffins in the morning and 36 muffins in the afternoon. "
        "If each muffin costs $3, how much money did the bakery make in total?"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch experiments for reasoning segmentation.")
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Single prompt. Can be repeated.",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default="",
        help="Path to .json list[str] or .txt prompts (one per line).",
    )
    parser.add_argument(
        "--segmenters",
        type=str,
        default="delimiter,keyword,hybrid,graph",
        help="Comma-separated list of segmenters to compare.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="torch",
        choices=["torch", "mlx"],
        help="Primary model backend.",
    )
    parser.add_argument(
        "--judge-backend",
        type=str,
        default="torch",
        choices=["torch", "mlx"],
        help="Judge model backend.",
    )
    parser.add_argument(
        "--primary-model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        help="Primary model ID.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Judge model ID (used for graph segmenter).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=700,
        help="Max generation tokens for primary model.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/experiments",
        help="Output directory for per-run and summary files.",
    )
    parser.add_argument(
        "--run-modes",
        type=str,
        default="baseline,mitigated",
        help="Comma-separated modes to compare: baseline and/or mitigated.",
    )
    parser.add_argument("--enable-diagnostics", action="store_true", help="Enable cognitive + DAG diagnostics.")
    parser.add_argument("--enable-latent", action="store_true", help="Enable latent loop classification.")
    parser.add_argument("--enable-pruning", action="store_true", help="Enable DAG pruning mitigation.")
    parser.add_argument("--enable-first-step-filter", action="store_true", help="Enable first-step quality scoring.")
    parser.add_argument("--enable-ccd", action="store_true", help="Enable CCD logits intervention.")
    return parser.parse_args()


def load_prompts(args: argparse.Namespace) -> List[str]:
    prompts: List[str] = [prompt.strip() for prompt in args.prompt if prompt.strip()]
    if args.prompts_file:
        path = Path(args.prompts_file)
        if not path.exists():
            raise FileNotFoundError(f"Prompts file does not exist: {path}")
        if path.suffix.lower() == ".json":
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(loaded, list):
                raise ValueError("JSON prompts file must be a list of strings.")
            prompts.extend(str(item).strip() for item in loaded if str(item).strip())
        else:
            lines = path.read_text(encoding="utf-8").splitlines()
            prompts.extend(line.strip() for line in lines if line.strip())
    if not prompts:
        prompts = [build_default_prompt()]
    return prompts


def aggregate_runs(runs: Sequence[ExperimentRun]) -> Dict[str, Any]:
    by_key: Dict[str, List[ExperimentRun]] = {}
    for run in runs:
        key = f"{run.run_mode}:{run.segmenter}"
        by_key.setdefault(key, []).append(run)

    summary: Dict[str, Any] = {"groups": {}, "overall_runs": len(runs)}
    for key, rows in by_key.items():
        unit_counts = [row.unit_count for row in rows]
        repeated = [row.repeated_unit_ratio for row in rows]
        overlaps = [row.adjacent_overlap_ratio for row in rows]
        loop_counts = [row.loop_count for row in rows]
        self_transition = [row.cognitive_self_transition_rate for row in rows]
        prune_ratios = [row.prune_ratio for row in rows]
        first_scores = [row.first_step_score for row in rows]
        loop_detected = [row.loop_detected for row in rows]
        judge_total = int(sum(row.judge_total_count for row in rows))
        judge_insert = int(sum(row.judge_insert_count for row in rows))
        judge_merge = int(sum(row.judge_merge_count for row in rows))
        latent_counts: Dict[str, int] = {}
        for row in rows:
            latent_counts[row.latent_classification] = latent_counts.get(row.latent_classification, 0) + 1

        summary["groups"][key] = {
            "runs": len(rows),
            "avg_unit_count": float(np.mean(unit_counts)) if unit_counts else 0.0,
            "avg_repeated_unit_ratio": float(np.mean(repeated)) if repeated else 0.0,
            "avg_adjacent_overlap_ratio": float(np.mean(overlaps)) if overlaps else 0.0,
            "avg_loop_count": float(np.mean(loop_counts)) if loop_counts else 0.0,
            "avg_cognitive_self_transition_rate": float(np.mean(self_transition)) if self_transition else 0.0,
            "avg_prune_ratio": float(np.mean(prune_ratios)) if prune_ratios else 0.0,
            "avg_first_step_score": float(np.mean(first_scores)) if first_scores else 0.0,
            "loop_detected_rate": float(np.mean(loop_detected)) if loop_detected else 0.0,
            "latent_classification_counts": latent_counts,
            "judge_insert_total": judge_insert,
            "judge_merge_total": judge_merge,
            "judge_insert_rate": float(judge_insert / judge_total) if judge_total else 0.0,
            "judge_merge_rate": float(judge_merge / judge_total) if judge_total else 0.0,
        }
    return summary


def main() -> None:
    configure_logging()
    args = parse_args()
    prompts = load_prompts(args)
    segmenters = [item.strip().lower() for item in args.segmenters.split(",") if item.strip()]
    run_modes = [item.strip().lower() for item in args.run_modes.split(",") if item.strip()]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    primary_model = ReasoningModel(
        ReasoningModelConfig(
            model_name_or_path=args.primary_model,
            max_new_tokens=args.max_new_tokens,
            temperature=0.6,
            use_4bit=True,
            backend=args.backend,
            device_preference="mps",
        )
    )

    judge_model = None
    needs_judge = (
        "graph" in segmenters
        or args.enable_diagnostics
        or args.enable_first_step_filter
    )
    if needs_judge:
        judge_model = ReasoningModel(
            ReasoningModelConfig(
                model_name_or_path=args.judge_model,
                max_new_tokens=120,
                temperature=0.1,
                use_4bit=True,
                backend=args.judge_backend,
                device_preference="mps",
            )
        )

    runs: List[ExperimentRun] = []
    for prompt_id, prompt in enumerate(prompts):
        for run_mode in run_modes:
            mode_is_mitigated = run_mode == "mitigated"
            for segmenter_name in segmenters:
                LOGGER.info(
                    "Running prompt_id=%s mode=%s segmenter=%s",
                    prompt_id,
                    run_mode,
                    segmenter_name,
                )
                segmenter = build_segmenter(
                    segmenter_name,
                    judge_model=judge_model,
                    triggers=["Alternatively", "Wait", "Hmm", "But wait"],
                )
                cognitive_analyzer = (
                    CognitiveFlowAnalyzer(judge_model)
                    if (judge_model and args.enable_diagnostics and mode_is_mitigated)
                    else None
                )
                graph_constructor = (
                    GraphConstructor(judge_model)
                    if (judge_model and args.enable_diagnostics and mode_is_mitigated)
                    else None
                )
                latent_monitor = LatentMonitor() if (args.enable_latent and mode_is_mitigated) else None
                pruning_engine = PruningEngine() if (args.enable_pruning and mode_is_mitigated) else None
                first_step_filter = (
                    FirstStepFilter(judge_model)
                    if (judge_model and args.enable_first_step_filter and mode_is_mitigated)
                    else None
                )
                ccd_processor = (
                    ConfusionContrastiveLogitsProcessor()
                    if (args.enable_ccd and mode_is_mitigated and args.backend == "torch")
                    else None
                )
                pipeline = ReasoningPipeline(
                    reasoning_model=primary_model,
                    segmenter=segmenter,
                    cognitive_analyzer=cognitive_analyzer,
                    graph_constructor=graph_constructor,
                    latent_monitor=latent_monitor,
                    pruning_engine=pruning_engine,
                    first_step_filter=first_step_filter,
                    ccd_logits_processor=ccd_processor,
                    enable_self_brake=mode_is_mitigated,
                )
                result = pipeline.run_with_details(prompt, return_hidden_states=False)

                loop_meta = result.metadata.get("loop_analysis", {})
                cognitive_meta = result.metadata.get("cognitive_flow", {})
                dag_meta = result.metadata.get("dag", {})
                latent_meta = result.metadata.get("latent") or {}
                first_step_meta = result.metadata.get("first_step_filter", {})
                decision_stats = (
                    segmenter.decision_stats()
                    if hasattr(segmenter, "decision_stats")
                    else {"insert": 0, "merge": 0, "total": 0}
                )
                node_count = int(dag_meta.get("node_count", 0))
                pruned_count = int(dag_meta.get("pruned_node_count", node_count))
                prune_ratio = 0.0
                if node_count > 0:
                    prune_ratio = float(max(0.0, (node_count - pruned_count) / node_count))

                runs.append(
                    ExperimentRun(
                        prompt_id=prompt_id,
                        run_mode=run_mode,
                        segmenter=segmenter_name,
                        unit_count=len(result.units),
                        repeated_unit_ratio=float(loop_meta.get("repeated_unit_ratio", 0.0)),
                        adjacent_overlap_ratio=float(loop_meta.get("adjacent_overlap_ratio", 0.0)),
                        loop_count=len(loop_meta.get("loop_indices", [])),
                        cognitive_self_transition_rate=float(cognitive_meta.get("self_transition_rate") or 0.0),
                        dag_node_count=node_count,
                        dag_pruned_node_count=pruned_count,
                        prune_ratio=prune_ratio,
                        latent_classification=str(latent_meta.get("classification", "UNKNOWN")),
                        first_step_score=float(first_step_meta.get("score") or 0.0),
                        loop_detected=1 if result.metadata.get("loop_detected", False) else 0,
                        judge_insert_count=int(decision_stats.get("insert", 0)),
                        judge_merge_count=int(decision_stats.get("merge", 0)),
                        judge_total_count=int(decision_stats.get("total", 0)),
                        units=[unit.text for unit in result.units],
                        metadata=result.metadata,
                    )
                )

    summary = aggregate_runs(runs)

    per_run_path = output_dir / "per_run.jsonl"
    with per_run_path.open("w", encoding="utf-8") as fp:
        for row in runs:
            fp.write(json.dumps(asdict(row), ensure_ascii=True) + "\n")

    summary_path = output_dir / "summary.json"
    summary_payload = {
        "config": {
            "primary_model": args.primary_model,
            "judge_model": args.judge_model,
            "backend": args.backend,
            "judge_backend": args.judge_backend,
            "segmenters": segmenters,
            "run_modes": run_modes,
            "enable_diagnostics": args.enable_diagnostics,
            "enable_latent": args.enable_latent,
            "enable_pruning": args.enable_pruning,
            "enable_first_step_filter": args.enable_first_step_filter,
            "enable_ccd": args.enable_ccd,
            "prompt_count": len(prompts),
        },
        "summary": summary,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    LOGGER.info("Wrote %s", per_run_path)
    LOGGER.info("Wrote %s", summary_path)


if __name__ == "__main__":
    main()
