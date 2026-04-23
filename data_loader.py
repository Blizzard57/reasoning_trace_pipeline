"""data_loader.py
Dataset loading for reasoning trace generation.

Supports:
  - aime24 / aime25   : local JSONL (from self_correction_llms data dir)
  - humaneval         : HuggingFace openai_humaneval
  - gpqa              : local JSONL (multiple-choice science)
  - hmmt              : local JSONL (competition math)

The reference repo (self_correction_llms) already has clean JSONL for
aime24/25, gpqa, hmmt — we reuse those directly rather than re-fetching.
HumanEval is fetched from HuggingFace (small, 164 examples).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# ---------------------------------------------------------------------------
# Prompt templates — mirroring the reference repo's style but extended to
# support every task type and model family we care about.
# ---------------------------------------------------------------------------
PROMPT_TEMPLATES: Dict[str, str] = {
    # DeepSeek-R1 distilled models (math / competition)
    "deepseek-r1-math": (
        "<｜begin▁of▁sentence｜>Please reason step by step, "
        "and put your final answer within \\boxed{{}}."
        "<｜User｜>{question}<｜Assistant｜><think>\n"
    ),
    # DeepSeek-R1 distilled — multiple choice (gpqa style)
    "deepseek-r1-choice": (
        "<｜begin▁of▁sentence｜>"
        "<｜User｜>Answer the following multiple choice question. "
        "The last line of your response should be of the following format: "
        "'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. "
        "Think step by step before answering.\n\n{question}<｜Assistant｜><think>\n"
    ),
    # DeepSeek-R1 distilled — coding
    "deepseek-r1-code": (
        "<｜begin▁of▁sentence｜>Please reason step by step. "
        "Write a complete Python solution inside a ```python``` block."
        "<｜User｜>{question}<｜Assistant｜><think>\n"
    ),
    # Gemma / generic chat models — uses apply_chat_template via mlx-lm,
    # so we just supply the user turn text and let the tokenizer wrap it.
    "generic-math": (
        "Please reason step by step and put your final answer within \\boxed{{}}.\n\n"
        "{question}"
    ),
    "generic-choice": (
        "Answer the following multiple choice question. "
        "The last line of your response should be of the following format: "
        "'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. "
        "Think step by step before answering.\n\n{question}"
    ),
    "generic-code": (
        "Write a complete Python solution for the following problem. "
        "Think step by step.\n\n{question}"
    ),
}

# ---------------------------------------------------------------------------
# Low-level JSONL helpers (matching the reference repo's utils/data.py)
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[data_loader] Skipping malformed line {i} in {path}: {exc}")
    return records


def _save_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[data_loader] Saved → {path}")


# ---------------------------------------------------------------------------
# Question formatters per dataset
# ---------------------------------------------------------------------------

def _format_choice_question(example: Dict[str, Any]) -> str:
    """Format MCQ: question + A/B/C/D options."""
    q = example.get("question", example.get("problem", ""))
    choices = example.get("choices", [])
    if not choices:
        return q.strip()
    opts = "\n".join(f"{letter}. {c}" for letter, c in zip("ABCD", choices))
    return f"{q.strip()}\n\n{opts}"


def _format_math_question(example: Dict[str, Any]) -> str:
    return example.get("question", example.get("problem", "")).strip()


def _format_code_question(example: Dict[str, Any]) -> str:
    """HumanEval: the 'prompt' field is already a complete docstring + signature."""
    return example.get("prompt", example.get("question", "")).strip()


def _parse_ground_truth(example: Dict[str, Any], dataset: str) -> str:
    """Return a plain-string ground truth for downstream eval."""
    if dataset in ("gpqa",):
        abcd = "ABCD"
        ans = example.get("answer", 0)
        if isinstance(ans, int):
            return abcd[ans]
        return str(ans)
    if dataset == "humaneval":
        return example.get("canonical_solution", "")
    # aime24, aime25, hmmt — numeric string
    return str(example.get("answer", "")).strip()

