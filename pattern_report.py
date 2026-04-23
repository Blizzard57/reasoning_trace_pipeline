"""
pattern_report.py
=================
Load analysis JSONL files and search for structural patterns in:
  - Where uninformative steps cluster (early / middle / late)
  - Whether redundant pairs are adjacent vs non-adjacent
  - Whether uninformative steps co-locate with redundant ones
  - Positional distribution of informative vs uninformative steps

Outputs a markdown + JSON report.

Usage:
    python pattern_report.py \\
        --analysis-dir outputs/analysis \\
        --output-dir outputs/reports
"""
from __future__ import annotations
import argparse, json, logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import statistics

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--analysis-dir", default="outputs/analysis")
    p.add_argument("--output-dir",   default="outputs/reports")
    p.add_argument("--dataset-filter", default="",
                   help="Only include files containing this substring.")
    return p.parse_args()


def _relative_pos(idx: int, n: int) -> str:
    """Bucket step index into early / middle / late thirds."""
    if n <= 1: return "early"
    frac = idx / (n - 1)
    if frac < 0.33:  return "early"
    if frac < 0.67:  return "middle"
    return "late"


def load_analysis(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]

def analyse_patterns(records: List[Dict]) -> Dict:
    """Extract structural patterns from a list of analysis records."""
    pos_counter:   Counter = Counter()   # "early"/"middle"/"late" for uninformative
    adj_redundant  = 0                   # redundant pairs where |i-j| == 1
    nonadj_redundant = 0                 # redundant pairs where |i-j| > 1
    uninform_then_inform = 0             # uninformative step immediately followed by informative
    inform_then_uninform = 0             # informative step immediately followed by uninformative
    step_label_seqs: List[List[str]] = []
    redundant_gap_counts: Counter = Counter()  # gap sizes for redundant pairs

    for rec in records:
        ana   = rec.get("analysis", {})
        n     = rec.get("n_steps", 0)
        if n == 0: continue

        lengths = ana.get("step_lengths", [])
        labels  = [ls.get("label", "marginal") for ls in lengths]
        step_label_seqs.append(labels)

        uninf_idx = ana.get("uninformative_idx", [])
        for idx in uninf_idx:
            pos_counter[_relative_pos(idx, n)] += 1

        # adjacent / non-adjacent uninformative runs
        for i in range(len(labels) - 1):
            if labels[i] == "uninformative" and labels[i+1] == "informative":
                uninform_then_inform += 1
            if labels[i] == "informative" and labels[i+1] == "uninformative":
                inform_then_uninform += 1

        for pair in ana.get("redundant_pairs", []):
            i, j = int(pair[0]), int(pair[1])
            gap = j - i
            redundant_gap_counts[gap] += 1
            if gap == 1:
                adj_redundant += 1
            else:
                nonadj_redundant += 1

    # Most common step label sequences (bigrams)
    bigram_counter: Counter = Counter()
    for seq in step_label_seqs:
        for k in range(len(seq) - 1):
            bigram_counter[(seq[k], seq[k+1])] += 1

    total_uninf = sum(pos_counter.values())
    return {
        "uninformative_position_distribution": {
            "early":  pos_counter.get("early",  0),
            "middle": pos_counter.get("middle", 0),
            "late":   pos_counter.get("late",   0),
            "total":  total_uninf,
            "pct_early":  round(pos_counter.get("early",0) / max(total_uninf,1), 3),
            "pct_middle": round(pos_counter.get("middle",0) / max(total_uninf,1), 3),
            "pct_late":   round(pos_counter.get("late",0) / max(total_uninf,1), 3),
        },
        "redundant_pairs": {
            "adjacent":     adj_redundant,
            "non_adjacent": nonadj_redundant,
            "gap_distribution": dict(sorted(redundant_gap_counts.items())),
        },
        "transition_bigrams": {
            f"{a}->{b}": cnt
            for (a, b), cnt in bigram_counter.most_common(20)
        },
        "uninform_then_inform_transitions": uninform_then_inform,
        "inform_then_uninform_transitions": inform_then_uninform,
    }

def render_markdown(patterns: Dict, source_info: str) -> str:
    p = patterns
    pos = p["uninformative_position_distribution"]
    rp  = p["redundant_pairs"]
    lines = [
        f"# Reasoning Trace Pattern Report",
        f"\n**Source:** {source_info}\n",
        "## 1. Where uninformative steps appear",
        f"| Position | Count | Fraction |",
        f"|----------|-------|----------|",
        f"| Early (0–33%) | {pos['early']} | {pos['pct_early']:.1%} |",
        f"| Middle (33–67%) | {pos['middle']} | {pos['pct_middle']:.1%} |",
        f"| Late (67–100%) | {pos['late']} | {pos['pct_late']:.1%} |",
        f"\n**Interpretation:** If early >> late, models front-load hedging.",
        f"If late >> early, models over-verify near the answer.\n",
        "## 2. Redundant step gap distribution",
        f"| Gap (|i-j|) | Count |",
        f"|------------|-------|",
    ]
    for gap, cnt in sorted(rp["gap_distribution"].items()):
        lines.append(f"| {gap} | {cnt} |")
    lines += [
        f"\n**Adjacent (gap=1):** {rp['adjacent']}  |  "
        f"**Non-adjacent:** {rp['non_adjacent']}",
        "\nIf gap=1 dominates: adjacent step pairs are often paraphrases.",
        "If gap>1 is common: the model revisits earlier ideas much later.\n",
        "## 3. Step-label transition bigrams (top 20)",
        f"| Transition | Count |",
        f"|------------|-------|",
    ]
    for trans, cnt in list(p["transition_bigrams"].items())[:20]:
        lines.append(f"| {trans} | {cnt} |")
    lines += [
        f"\n**uninformative→informative:** {p['uninform_then_inform_transitions']}",
        f"**informative→uninformative:** {p['inform_then_uninform_transitions']}",
        "\nHigh uninformative→informative count suggests brief \"pivot\" phrases",
        "before substantive reasoning steps — a common model habit.\n",
    ]
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    analysis_dir = Path(args.analysis_dir)
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(analysis_dir.glob("*_analysis.jsonl"))
    if args.dataset_filter:
        files = [f for f in files if args.dataset_filter in f.name]
    if not files:
        LOGGER.error("No *_analysis.jsonl files in %s", analysis_dir); return

    all_records: List[Dict] = []
    for fpath in files:
        LOGGER.info("Loading %s", fpath.name)
        all_records.extend(load_analysis(fpath))

    patterns = analyse_patterns(all_records)
    source_info = ", ".join(f.name for f in files)
    md = render_markdown(patterns, source_info)

    (output_dir / "patterns.md").write_text(md, encoding="utf-8")
    (output_dir / "patterns.json").write_text(
        json.dumps(patterns, indent=2), encoding="utf-8")
    LOGGER.info("Written to %s", output_dir)
    print(md)


if __name__ == "__main__":
    main()
