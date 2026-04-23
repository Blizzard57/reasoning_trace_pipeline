from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from models import ReasoningModel

LOGGER = logging.getLogger(__name__)


def _extract_answer_text(result: Any) -> str:
    """Return the post-think answer text from a GenerationResult.

    Reasoning models wrap their chain-of-thought in <think>...</think> and emit
    the final answer *after* the closing tag.  Using ``result.think_text`` gives
    the internal scratchpad, which contains raw LaTeX / math and never the
    structured label we asked for.  Instead we slice ``full_text`` at
    ``</think>`` and return whatever follows; if the tag is absent we fall back
    to ``full_text`` directly.
    """
    full = (result.full_text or "").strip()
    marker = "</think>"
    idx = full.lower().rfind(marker.lower())
    if idx != -1:
        return full[idx + len(marker):].strip()
    return full


COGNITIVE_LABELS: Tuple[str, ...] = (
    "ProblemSetup",
    "Calculation",
    "Verification",
    "Interpretation",
    "Conclusion",
    "Other",
)


@dataclass
class LabeledUnit:
    label: str
    rationale: str


class _UnitLike(Protocol):
    text: str
    cognitive_state: Optional[str]


class CognitiveFlowAnalyzer:
    """
    Uses a judge model to label each step with a functional cognitive role.
    Intended to detect state-transition loops (e.g., repeated Verification).
    """

    def __init__(self, judge_model: ReasoningModel) -> None:
        self.judge_model = judge_model

    def label_units(self, units: Sequence[_UnitLike]) -> List[LabeledUnit]:
        labeled: List[LabeledUnit] = []
        for unit in units:
            labeled.append(self._label_one(unit.text))
        return labeled

    def state_transition_matrix(self, labels: Sequence[str]) -> np.ndarray:
        idx = {label: i for i, label in enumerate(COGNITIVE_LABELS)}
        mat = np.zeros((len(COGNITIVE_LABELS), len(COGNITIVE_LABELS)), dtype=np.float32)
        for i in range(1, len(labels)):
            a = idx.get(labels[i - 1], idx["Other"])
            b = idx.get(labels[i], idx["Other"])
            mat[a, b] += 1.0
        return mat

    def self_transition_rate(self, labels: Sequence[str]) -> float:
        if len(labels) < 2:
            return 0.0
        same = sum(1 for i in range(1, len(labels)) if labels[i] == labels[i - 1])
        return float(same / (len(labels) - 1))

    def _label_one(self, text: str) -> LabeledUnit:
        prompt = (
            "You are a cognitive-flow judge.\n"
            "Assign exactly one label to the STEP from this set:\n"
            f"{', '.join(COGNITIVE_LABELS)}\n\n"
            "Rules:\n"
            "- Reply with ONLY a single line in this exact format: <label>|<rationale>\n"
            "- <label> must be one of the listed labels, copied exactly.\n"
            "- Do not include any other text, explanation, or thinking in your reply.\n\n"
            "Example output: Calculation|The step computes a numeric value.\n\n"
            f"STEP:\n{text}\n"
        )
        result = self.judge_model.generate(prompt, return_hidden_states=False)
        raw = _extract_answer_text(result).strip()
        # Scan all lines bottom-up for the first that contains a valid label
        candidate_line = ""
        for line in reversed(raw.splitlines()):
            line = line.strip()
            if not line:
                continue
            candidate_line = line
            if "|" in line:
                potential_label = line.split("|", 1)[0].strip()
                if potential_label in COGNITIVE_LABELS:
                    break
        first = candidate_line if candidate_line else "Other|Empty judge output"
        if "|" in first:
            label, rationale = first.split("|", 1)
        else:
            label, rationale = first, "No explicit rationale."

        label = label.strip()
        if label not in COGNITIVE_LABELS:
            LOGGER.info("Unrecognized label '%s'; mapping to Other.", label)
            label = "Other"
        return LabeledUnit(label=label, rationale=rationale.strip())


@dataclass
class GraphNode:
    node_id: int
    text: str
    role: str  # Progress | Review
    label: Optional[str] = None
    depth: float = 0.0
    parents: List[int] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.parents is None:
            self.parents = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ReasoningDAG:
    nodes: List[GraphNode]
    edges: List[Tuple[int, int]]  # (src, dst)

    def descendants_count(self) -> Dict[int, int]:
        children: Dict[int, List[int]] = {n.node_id: [] for n in self.nodes}
        for src, dst in self.edges:
            children.setdefault(src, []).append(dst)

        memo: Dict[int, int] = {}

        def dfs(u: int) -> int:
            if u in memo:
                return memo[u]
            total = 0
            for v in children.get(u, []):
                total += 1 + dfs(v)
            memo[u] = total
            return total

        return {n.node_id: dfs(n.node_id) for n in self.nodes}


class GraphConstructor:
    """
    Builds a simple DAG over a segmented reasoning trace.
    Nodes are labeled Progress vs Review.

    Practical choice for now: enforce acyclicity by only allowing edges from
    earlier -> later nodes; Review nodes attach to most recent Progress node.
    """

    def __init__(self, judge_model: ReasoningModel) -> None:
        self.judge_model = judge_model

    def build(self, units: Sequence[_UnitLike]) -> ReasoningDAG:
        nodes: List[GraphNode] = []
        edges: List[Tuple[int, int]] = []
        last_progress_id: Optional[int] = None

        for i, unit in enumerate(units):
            depth = float(i / max(1, len(units) - 1))
            role, rationale = self._progress_vs_review(unit.text)
            node = GraphNode(
                node_id=i,
                text=unit.text,
                role=role,
                label=unit.cognitive_state,
                depth=depth,
                parents=[],
                metadata={"rationale": rationale},
            )
            nodes.append(node)

            if i == 0:
                if role == "Progress":
                    last_progress_id = 0
                continue

            if role == "Review" and last_progress_id is not None:
                edges.append((last_progress_id, i))
                node.parents.append(last_progress_id)
            else:
                edges.append((i - 1, i))
                node.parents.append(i - 1)

            if role == "Progress":
                last_progress_id = i

        return ReasoningDAG(nodes=nodes, edges=edges)

    def _progress_vs_review(self, text: str) -> Tuple[str, str]:
        prompt = (
            "You are a reasoning-graph judge.\n"
            "Classify this step as:\n"
            "- Progress: advances the solution with new information or computation\n"
            "- Review: re-checks, rephrases, or validates prior work without adding substance\n\n"
            "Rules:\n"
            "- Reply with ONLY a single line in this exact format: <role>|<rationale>\n"
            "- <role> must be exactly Progress or Review.\n"
            "- Do not include any other text, explanation, or thinking in your reply.\n\n"
            "Example output: Progress|Introduces a new calculation step.\n\n"
            f"STEP:\n{text}\n"
        )
        result = self.judge_model.generate(prompt, return_hidden_states=False)
        raw = _extract_answer_text(result).strip()
        # Scan lines bottom-up for first with a valid role
        candidate_line = ""
        for line in reversed(raw.splitlines()):
            line = line.strip()
            if not line:
                continue
            candidate_line = line
            if "|" in line:
                potential_role = line.split("|", 1)[0].strip().capitalize()
                if potential_role in {"Progress", "Review"}:
                    break
        first = candidate_line if candidate_line else "Review|Empty judge output"
        if "|" in first:
            role, rationale = first.split("|", 1)
        else:
            role, rationale = first, "No explicit rationale."
        role = role.strip().capitalize()
        if role not in {"Progress", "Review"}:
            role = "Review"
        return role, rationale.strip()

