from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from models import ReasoningModel, ReasoningModelConfig
from pipeline import ReasoningPipeline
from diagnostics import CognitiveFlowAnalyzer, GraphConstructor
from mechanistic import LatentMonitor
from mitigation import ConfusionContrastiveLogitsProcessor, FirstStepFilter, PruningEngine
from segmentation import build_segmenter


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def build_gsm8k_prompt() -> str:
    return (
        "Solve this math word problem and think step by step.\n"
        "Use <think>...</think> tags for your reasoning and provide a final answer.\n\n"
        "A bakery sold 48 muffins in the morning and 36 muffins in the afternoon. "
        "If each muffin costs $3, how much money did the bakery make in total?"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reasoning trace segmentation pipeline.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Custom prompt. If omitted, uses built-in GSM8K-style prompt.",
    )
    parser.add_argument(
        "--segmenter",
        type=str,
        default="graph",
        choices=["delimiter", "keyword", "hybrid", "graph"],
        help="Segmentation strategy.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="torch",
        choices=["torch", "mlx"],
        help="Primary model inference backend.",
    )
    parser.add_argument(
        "--judge-backend",
        type=str,
        default="torch",
        choices=["torch", "mlx"],
        help="Judge model backend (only used for graph segmenter).",
    )
    parser.add_argument(
        "--primary-model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        help="HF or MLX model identifier for primary reasoning model.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HF or MLX model identifier for judge model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=700,
        help="Max new tokens for generation.",
    )
    parser.add_argument(
        "--return-hidden-states",
        action="store_true",
        help="Return hidden states from final generation step (torch backend).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional path to save full run result as JSON.",
    )
    parser.add_argument(
        "--enable-diagnostics",
        action="store_true",
        help="Enable cognitive flow labeling and DAG construction (requires judge model).",
    )
    parser.add_argument(
        "--enable-latent",
        action="store_true",
        help="Enable latent loop monitoring (torch backend exposes trajectories).",
    )
    parser.add_argument(
        "--enable-pruning",
        action="store_true",
        help="Enable graph pruning mitigation (requires DAG).",
    )
    parser.add_argument(
        "--enable-first-step-filter",
        action="store_true",
        help="Enable PRM-like first-step scoring (requires judge model).",
    )
    parser.add_argument(
        "--enable-ccd",
        action="store_true",
        help="Enable CCD-style logits processing during decoding (torch backend).",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    primary_config = ReasoningModelConfig(
        model_name_or_path=args.primary_model,
        max_new_tokens=args.max_new_tokens,
        temperature=0.6,
        use_4bit=True,
        backend=args.backend,
        device_preference="mps",
    )

    reasoning_model = ReasoningModel(primary_config)
    judge_model = None
    needs_judge = args.segmenter == "graph" or args.enable_diagnostics or args.enable_first_step_filter
    if needs_judge:
        judge_config = ReasoningModelConfig(
            model_name_or_path=args.judge_model,
            max_new_tokens=120,
            temperature=0.1,
            use_4bit=True,
            backend=args.judge_backend,
            device_preference="mps",
        )
        judge_model = ReasoningModel(judge_config)
    segmenter = build_segmenter(
        args.segmenter,
        judge_model=judge_model,
        triggers=["Alternatively", "Wait", "Hmm", "But wait"],
    )

    cognitive_analyzer = CognitiveFlowAnalyzer(judge_model) if (args.enable_diagnostics and judge_model) else None
    graph_constructor = GraphConstructor(judge_model) if (args.enable_diagnostics and judge_model) else None
    latent_monitor = LatentMonitor() if args.enable_latent else None
    pruning_engine = PruningEngine() if args.enable_pruning else None
    first_step_filter = FirstStepFilter(judge_model) if (args.enable_first_step_filter and judge_model) else None
    ccd_processor = ConfusionContrastiveLogitsProcessor() if args.enable_ccd else None

    pipeline = ReasoningPipeline(
        reasoning_model=reasoning_model,
        segmenter=segmenter,
        cognitive_analyzer=cognitive_analyzer,
        graph_constructor=graph_constructor,
        latent_monitor=latent_monitor,
        pruning_engine=pruning_engine,
        first_step_filter=first_step_filter,
        ccd_logits_processor=ccd_processor,
        enable_self_brake=True,
    )
    prompt = args.prompt or build_gsm8k_prompt()

    detailed = pipeline.run_with_details(
        prompt,
        return_hidden_states=args.return_hidden_states,
    )
    print("\n=== Atomic Reasoning Units ===")
    for unit in detailed.units:
        print(f"[{unit.index}] {unit.text}\n")

    print("=== Run Metadata ===")
    print(detailed.metadata)
    if detailed.hidden_states is not None:
        print(f"Hidden states tensor shape: {tuple(detailed.hidden_states.shape)}")
    else:
        print("Hidden states not returned.")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "prompt": prompt,
            "units": [
                {
                    "index": unit.index,
                    "text": unit.text,
                    "cognitive_state": unit.cognitive_state,
                    "metadata": unit.metadata,
                }
                for unit in detailed.units
            ],
            "full_text": detailed.full_text,
            "think_text": detailed.think_text,
            "metadata": detailed.metadata,
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved run result to: {output_path}")


if __name__ == "__main__":
    main()
