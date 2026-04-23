from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import torch
from transformers import LogitsProcessor

from diagnostics import ReasoningDAG
from models import ReasoningModel

LOGGER = logging.getLogger(__name__)


def _extract_answer_text(result: Any) -> str:
    """Return post-think answer text from a GenerationResult."""
    full = (result.full_text or "").strip()
    marker = "</think>"
    idx = full.lower().rfind(marker.lower())
    if idx != -1:
        return full[idx + len(marker):].strip()
    return full


class AtomicUnitLike(Protocol):
    text: str
    index: int
    cognitive_state: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class PruningConfig:
    branch_descendant_k: int = 2
    depth_threshold_m: float = 0.9


class PruningEngine:
    """
    Graph pruning strategies for removing redundant Review nodes.
    """

    def __init__(self, config: PruningConfig | None = None) -> None:
        self.config = config or PruningConfig()

    def prune(self, dag: ReasoningDAG) -> Tuple[ReasoningDAG, Dict[str, Any]]:
        descendants = dag.descendants_count()
        keep = {n.node_id for n in dag.nodes}

        removed_branch: List[int] = []
        removed_depth: List[int] = []

        for node in dag.nodes:
            if node.role != "Review":
                continue
            if descendants.get(node.node_id, 0) < self.config.branch_descendant_k:
                keep.discard(node.node_id)
                removed_branch.append(node.node_id)

        for node in dag.nodes:
            if node.node_id not in keep:
                continue
            if node.role == "Review" and node.depth >= self.config.depth_threshold_m:
                keep.discard(node.node_id)
                removed_depth.append(node.node_id)

        new_nodes = [n for n in dag.nodes if n.node_id in keep]
        new_edges = [(s, d) for (s, d) in dag.edges if s in keep and d in keep]
        meta = {
            "removed_branch_nodes": removed_branch,
            "removed_depth_nodes": removed_depth,
            "kept_nodes": sorted(list(keep)),
        }
        return ReasoningDAG(nodes=new_nodes, edges=new_edges), meta

    def prune_units_from_dag(
        self,
        units: Sequence[AtomicUnitLike],
        dag: ReasoningDAG,
    ) -> List[AtomicUnitLike]:
        keep_ids = {n.node_id for n in dag.nodes}
        kept = [u for u in units if u.index in keep_ids]
        # re-index to keep atomic ordering consistent
        remapped: List[AtomicUnitLike] = []
        for new_i, unit in enumerate(kept):
            try:
                setattr(unit, "index", new_i)
            except Exception:
                pass
            remapped.append(unit)
        return remapped


class ConfusionContrastiveLogitsProcessor(LogitsProcessor):
    """
    Simple CCD-style intervention:
    - If max softmax prob < tau, subtract alpha * "confused" distribution.
    - Confused distribution is made by masking top-k anchor tokens and renormalizing.

    This is a lightweight approximation meant for ablation-style research.
    """

    def __init__(self, tau: float = 0.35, alpha: float = 0.8, top_k: int = 32) -> None:
        self.tau = tau
        self.alpha = alpha
        self.top_k = top_k

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        probs = torch.softmax(scores, dim=-1)
        max_prob = probs.max(dim=-1).values  # (batch,)
        if torch.all(max_prob >= self.tau):
            return scores

        # Build confused distribution by masking top-k anchors.
        topk = torch.topk(probs, k=min(self.top_k, probs.shape[-1]), dim=-1).indices
        confused = probs.clone()
        confused.scatter_(dim=-1, index=topk, value=0.0)
        confused = confused / (confused.sum(dim=-1, keepdim=True) + 1e-8)

        # Subtract in logit space (approx via log).
        adjustment = self.alpha * torch.log(confused + 1e-8)
        new_scores = scores - adjustment
        return new_scores


@dataclass
class FirstStepFilterConfig:
    min_score: float = 0.35


class FirstStepFilter:
    """
    PRM-like early filter using a judge model.
    If score is low, caller can discard the whole path early.
    """

    def __init__(self, judge_model: ReasoningModel, config: FirstStepFilterConfig | None = None) -> None:
        self.judge_model = judge_model
        self.config = config or FirstStepFilterConfig()

    def score_first_step(self, text: str) -> Tuple[float, str]:
        prompt = (
            "You are a reward model for first-step quality.\n"
            "Score the STEP from 0.0 to 1.0 based on whether it sets up the solution well.\n"
            "Rules:\n"
            "- Reply with ONLY a single line in this exact format: <score>|<rationale>\n"
            "- <score> must be a decimal number between 0.0 and 1.0.\n"
            "- Do not include any other text, explanation, or thinking in your reply.\n\n"
            "Example output: 0.8|The step clearly identifies the known values.\n\n"
            f"STEP:\n{text}\n"
        )
        result = self.judge_model.generate(prompt, return_hidden_states=False)
        raw = _extract_answer_text(result).strip()
        first = ""
        for line in reversed(raw.splitlines()):
            line = line.strip()
            if line:
                first = line
                break
        if not first:
            first = "0.0|Empty output"
        if "|" in first:
            score_str, rationale = first.split("|", 1)
        else:
            score_str, rationale = first, "No rationale."
        try:
            score = float(score_str.strip())
        except Exception:
            score = 0.0
            rationale = f"Unparseable score; raw={first}"
        score = max(0.0, min(1.0, score))
        return score, rationale.strip()

    def should_keep(self, score: float) -> bool:
        return score >= self.config.min_score


def epiphany_sentence() -> str:
    return "Wait — I've checked this already. Time to move forward with the solution."

