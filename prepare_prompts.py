from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract prompts using self_correction_llms prompt templates."
    )
    parser.add_argument(
        "--source-root",
        type=str,
        default="/Users/blizzard/Documents/Projects/self_correction_llms",
        help="Path to self_correction_llms project root.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="",
        help="Optional data directory override. Default: <source-root>/data",
    )
    parser.add_argument(
        "--data-names",
        type=str,
        default="gsm8k",
        help="Comma-separated dataset names (e.g., gsm8k,math).",
    )
    parser.add_argument("--split", type=str, default="test", help="Dataset split.")
    parser.add_argument(
        "--prompt-type",
        type=str,
        default="deepseek-r1",
        help="Prompt template key from self_correction_llms utils/data.py",
    )
    parser.add_argument(
        "--num-test-sample",
        type=int,
        default=-1,
        help="-1 for full data, else first N examples.",
    )
    parser.add_argument("--start", type=int, default=0, help="Start index.")
    parser.add_argument("--end", type=int, default=-1, help="-1 for end, else end index.")
    parser.add_argument(
        "--output-json",
        type=str,
        default="prompts.json",
        help="Output path for prompt list (JSON list[str]).",
    )
    parser.add_argument(
        "--output-records-json",
        type=str,
        default="",
        help="Optional output path for detailed records (idx/question/prompt).",
    )
    parser.add_argument(
        "--on-missing",
        type=str,
        default="error",
        choices=["error", "skip"],
        help="Behavior when a dataset is missing.",
    )
    return parser.parse_args()


def load_jsonl(file_path: Path) -> Iterable[Dict[str, Any]]:
    with file_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def parse_question(example: Dict[str, Any], data_name: str) -> str:
    # Mirrors self_correction_llms/utils/parser.py logic without math_verify dependency.
    if data_name in {"mmlu_stem", "gpqa"}:
        options = list(example["choices"])
        if len(options) != 4:
            return ""
        for i, (label, option) in enumerate(zip("ABCD", options)):
            options[i] = f"{label}. {str(option).strip()}\n"
        joined = " ".join(options).strip()
        return f"{str(example['question']).strip()}\n\n {joined}".strip()

    for key in ("question", "problem", "Question", "input"):
        if key in example:
            return str(example[key]).strip()
    return ""


def load_prompt_templates(source_root: Path) -> Dict[str, Any]:
    data_py = source_root / "utils" / "data.py"
    if not data_py.exists():
        raise FileNotFoundError(f"Cannot find prompt template file: {data_py}")

    spec = importlib.util.spec_from_file_location("sc_data_module", str(data_py))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {data_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    templates = getattr(module, "PROMPT_TEMPLATES", None)
    if not isinstance(templates, dict):
        raise RuntimeError("PROMPT_TEMPLATES missing or invalid in source utils/data.py")
    return templates


def construct_prompt(question: str, prompt_type: str, prompt_templates: Dict[str, Any]) -> str:
    if prompt_type not in prompt_templates:
        available = ", ".join(sorted(prompt_templates.keys()))
        raise ValueError(f"Unknown prompt type '{prompt_type}'. Available: {available}")
    input_template = prompt_templates[prompt_type][0]
    args = SimpleNamespace(prompt_type=prompt_type)
    _ = args  # keep parity with source style; not used directly
    return input_template.format(input=question).strip(" ")


def load_dataset_examples(data_file: Path) -> List[Dict[str, Any]]:
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_file}")
    examples = list(load_jsonl(data_file))
    if not examples:
        return []
    if "idx" not in examples[0]:
        examples = [{"idx": i, **ex} for i, ex in enumerate(examples)]
    examples = sorted(examples, key=lambda x: x["idx"])
    return examples


def available_datasets(data_dir: Path, split: str) -> List[str]:
    if not data_dir.exists():
        return []
    names: List[str] = []
    for child in data_dir.iterdir():
        if not child.is_dir():
            continue
        if (child / f"{split}.jsonl").exists():
            names.append(child.name)
    return sorted(names)


def resolve_data_file(data_dir: Path, data_name_or_path: str, split: str) -> Path:
    as_path = Path(data_name_or_path)
    if as_path.suffix == ".jsonl":
        return as_path if as_path.is_absolute() else (Path.cwd() / as_path).resolve()
    return data_dir / data_name_or_path / f"{split}.jsonl"


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    data_dir = Path(args.data_dir) if args.data_dir else source_root / "data"
    prompt_templates = load_prompt_templates(source_root)

    all_prompts: List[str] = []
    all_records: List[Dict[str, Any]] = []
    discovered = available_datasets(data_dir, args.split)

    for data_name in [x.strip() for x in args.data_names.split(",") if x.strip()]:
        data_file = resolve_data_file(data_dir, data_name, args.split)
        if not data_file.exists():
            suggestion = difflib.get_close_matches(data_name, discovered, n=3)
            msg = (
                f"Dataset file not found: {data_file}\n"
                f"Requested data_name: {data_name}\n"
                f"Available datasets for split='{args.split}' under {data_dir}: "
                f"{', '.join(discovered) if discovered else '(none found)'}"
            )
            if suggestion:
                msg += f"\nDid you mean: {', '.join(suggestion)}"
            if args.on_missing == "skip":
                print(f"[warn] {msg}")
                continue
            raise FileNotFoundError(msg)

        examples = load_dataset_examples(data_file)
        if args.num_test_sample > 0:
            examples = examples[: args.num_test_sample]
        end_idx = len(examples) if args.end == -1 else args.end
        examples = examples[args.start : end_idx]

        for ex in examples:
            question = parse_question(ex, data_name)
            if not question:
                continue
            prompt = construct_prompt(question, args.prompt_type, prompt_templates)
            all_prompts.append(prompt)
            all_records.append(
                {
                    "data_name": data_name,
                    "idx": ex.get("idx"),
                    "question": question,
                    "prompt": prompt,
                }
            )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(all_prompts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(all_prompts)} prompts to {output_path}")

    if args.output_records_json:
        records_path = Path(args.output_records_json)
        records_path.parent.mkdir(parents=True, exist_ok=True)
        records_path.write_text(
            json.dumps(all_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote {len(all_records)} records to {records_path}")


if __name__ == "__main__":
    main()
