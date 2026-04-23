from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize baseline vs mitigated experiment deltas.")
    parser.add_argument(
        "--summary-json",
        type=str,
        default="outputs/exp_compare/summary.json",
        help="Path to experiments summary.json",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs/exp_compare/report.md",
        help="Path to write markdown report.",
    )
    return parser.parse_args()


def _get_metric(group: Dict[str, Any], key: str) -> float:
    value = group.get(key, 0.0)
    try:
        return float(value)
    except Exception:
        return 0.0


def _extract_pairs(groups: Dict[str, Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    out: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
    segmenters = set()
    for key in groups:
        if ":" not in key:
            continue
        _, segmenter = key.split(":", 1)
        segmenters.add(segmenter)

    for segmenter in sorted(segmenters):
        base = groups.get(f"baseline:{segmenter}", {})
        mitigated = groups.get(f"mitigated:{segmenter}", {})
        if base and mitigated:
            out.append((segmenter, base, mitigated))
    return out


def _delta(base: float, mitigated: float) -> float:
    return mitigated - base


def _score_improvement(base: Dict[str, Any], mitigated: Dict[str, Any]) -> float:
    # Lower is better for loop-like metrics.
    w_rep = 0.35
    w_overlap = 0.25
    w_self = 0.2
    w_loop = 0.2
    return (
        w_rep * (_get_metric(base, "avg_repeated_unit_ratio") - _get_metric(mitigated, "avg_repeated_unit_ratio"))
        + w_overlap
        * (_get_metric(base, "avg_adjacent_overlap_ratio") - _get_metric(mitigated, "avg_adjacent_overlap_ratio"))
        + w_self
        * (
            _get_metric(base, "avg_cognitive_self_transition_rate")
            - _get_metric(mitigated, "avg_cognitive_self_transition_rate")
        )
        + w_loop * (_get_metric(base, "loop_detected_rate") - _get_metric(mitigated, "loop_detected_rate"))
    )


def build_report(summary_payload: Dict[str, Any]) -> str:
    groups = (summary_payload.get("summary") or {}).get("groups") or {}
    pairs = _extract_pairs(groups)
    if not pairs:
        return (
            "# Experiment Delta Report\n\n"
            "No baseline/mitigated segmenter pairs were found.\n"
            "Ensure experiments were run with `--run-modes baseline,mitigated`.\n"
        )

    scored: List[Tuple[float, str, Dict[str, Any], Dict[str, Any]]] = []
    for segmenter, base, mitigated in pairs:
        scored.append((_score_improvement(base, mitigated), segmenter, base, mitigated))
    scored.sort(key=lambda x: x[0], reverse=True)

    lines: List[str] = []
    lines.append("# Experiment Delta Report")
    lines.append("")
    lines.append("Compares `baseline:*` vs `mitigated:*` groups per segmenter.")
    lines.append("")
    lines.append("## Ranking")
    lines.append("")
    for rank, (score, segmenter, _, _) in enumerate(scored, start=1):
        lines.append(f"{rank}. `{segmenter}` -> composite improvement score: `{score:.4f}`")
    lines.append("")
    lines.append("## Segmenter Deltas")
    lines.append("")

    metric_keys = [
        "avg_repeated_unit_ratio",
        "avg_adjacent_overlap_ratio",
        "avg_cognitive_self_transition_rate",
        "loop_detected_rate",
        "avg_first_step_score",
        "avg_prune_ratio",
    ]

    for _, segmenter, base, mitigated in scored:
        lines.append(f"### `{segmenter}`")
        for key in metric_keys:
            b = _get_metric(base, key)
            m = _get_metric(mitigated, key)
            d = _delta(b, m)
            lines.append(f"- `{key}`: baseline={b:.4f}, mitigated={m:.4f}, delta={d:+.4f}")
        lines.append(
            f"- `latent_classification_counts` baseline={base.get('latent_classification_counts', {})} "
            f"mitigated={mitigated.get('latent_classification_counts', {})}"
        )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_json)
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    report_md = build_report(payload)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_md, encoding="utf-8")
    print(f"Wrote report to {output_path}")


if __name__ == "__main__":
    main()
