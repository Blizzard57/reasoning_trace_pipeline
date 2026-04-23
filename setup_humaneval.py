"""
setup_humaneval.py
==================
Downloads HumanEval problems directly from the OpenAI HumanEval GitHub repo
(no `datasets` library required) and saves them as data/humaneval/test.jsonl.

Run once before generating traces:
    python setup_humaneval.py [--num-problems 20]
"""
from __future__ import annotations
import argparse, gzip, json, urllib.request
from pathlib import Path

# Raw JSONL.gz from the official OpenAI HumanEval release
HUMANEVAL_URL = (
    "https://raw.githubusercontent.com/openai/human-eval/master/"
    "data/HumanEval.jsonl.gz"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--num-problems", type=int, default=20,
                   help="How many HumanEval problems to save (default 20).")
    p.add_argument("--out-dir", default="",
                   help="Override output directory (default: <repo>/data/humaneval).")
    return p.parse_args()


def fetch_humaneval(num_problems: int) -> list:
    """Download and parse HumanEval JSONL.gz from GitHub."""
    print(f"Downloading HumanEval from:\n  {HUMANEVAL_URL}")
    with urllib.request.urlopen(HUMANEVAL_URL, timeout=30) as resp:
        raw = resp.read()
    lines = gzip.decompress(raw).decode("utf-8").splitlines()
    examples = [json.loads(l) for l in lines if l.strip()]
    print(f"  → {len(examples)} total problems found.")
    return examples[:num_problems]


def convert(ex: dict, idx: int) -> dict:
    """Convert a HumanEval record to our standard JSONL schema."""
    return {
        "idx":         idx,
        "task_id":     ex.get("task_id", f"HumanEval/{idx}"),
        # 'question' = the docstring prompt the model must reason about
        "question":    ex.get("prompt", ""),
        "answer":      ex.get("canonical_solution", ""),
        "entry_point": ex.get("entry_point", ""),
        "test":        ex.get("test", ""),
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).parent
    out_dir = Path(args.out_dir) if args.out_dir else repo_root / "data" / "humaneval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "test.jsonl"

    try:
        examples = fetch_humaneval(args.num_problems)
    except Exception as e:
        print(f"Download failed: {e}")
        print("Writing the 3-problem hardcoded fallback instead.")
        examples = _fallback()

    records = [convert(ex, i) for i, ex in enumerate(examples)]

    with out_file.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} problems to {out_file}")
    # Show first problem as a sanity check
    if records:
        print(f"\nFirst problem preview (task_id={records[0]['task_id']}):")
        print(records[0]["question"][:300], "…")


def _fallback() -> list:
    """10 hardcoded HumanEval problems — used only if network is unavailable."""
    return [
        {"task_id": "HumanEval/0",
         "prompt": "from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check whether in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
         "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n    return False\n",
         "entry_point": "has_close_elements", "test": ""},
        {"task_id": "HumanEval/1",
         "prompt": "from typing import List\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input is a string of nested parentheses. Return list of separated groups.\"\"\"\n",
         "canonical_solution": "    result = []\n    current_string = []\n    current_depth = 0\n    for c in paren_string:\n        if c == '(': current_depth += 1; current_string.append(c)\n        elif c == ')': current_depth -= 1; current_string.append(c)\n        if current_depth == 0 and current_string:\n            result.append(''.join(current_string)); current_string = []\n    return result\n",
         "entry_point": "separate_paren_groups", "test": ""},
        {"task_id": "HumanEval/2",
         "prompt": "def truncate_number(number: float) -> float:\n    \"\"\" Return decimal part of a positive float. \"\"\"\n",
         "canonical_solution": "    return number % 1.0\n",
         "entry_point": "truncate_number", "test": ""},
        {"task_id": "HumanEval/3",
         "prompt": "from typing import List\n\ndef below_zero(operations: List[int]) -> bool:\n    \"\"\" Return True if balance drops below zero at any point. \"\"\"\n",
         "canonical_solution": "    balance = 0\n    for op in operations:\n        balance += op\n        if balance < 0: return True\n    return False\n",
         "entry_point": "below_zero", "test": ""},
        {"task_id": "HumanEval/4",
         "prompt": "from typing import List\n\ndef mean_absolute_deviation(numbers: List[float]) -> float:\n    \"\"\" Return the Mean Absolute Deviation around the mean of the dataset. \"\"\"\n",
         "canonical_solution": "    mean = sum(numbers) / len(numbers)\n    return sum(abs(x - mean) for x in numbers) / len(numbers)\n",
         "entry_point": "mean_absolute_deviation", "test": ""},
    ]


if __name__ == "__main__":
    main()
