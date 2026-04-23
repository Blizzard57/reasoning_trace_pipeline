from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from models import ReasoningModel
from segmentation import BaseSegmenter

LOGGER = logging.getLogger(__name__)


@dataclass
class AtomicReasoningUnit:
    text: str
    index: int
    cognitive_state: Optional[str] = None
    label: Optional[str] = None
    role: Optional[str] = None  # Progress | Review
    depth: Optional[float] = None
    redundancy_score: Optional[float] = None
    latent_classification: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DualKVCache:
    """Placeholder structure for future contrastive decoding experiments."""

    main_cache: Optional[Any] = None
    neutral_cache: Optional[Any] = None


@dataclass
class PipelineRunResult:
    units: List[AtomicReasoningUnit]
    full_text: str
    think_text: str
    hidden_states: Optional[torch.Tensor] = None
    dual_kv_cache: DualKVCache = field(default_factory=DualKVCache)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoopAnalysis:
    """Heuristic loop signals for downstream filtering and experiments."""

    repeated_unit_ratio: float
    adjacent_overlap_ratio: float
    loop_indices: List[int] = field(default_factory=list)
    notes: Dict[str, Any] = field(default_factory=dict)


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def analyze_reasoning_loops(units: Sequence[AtomicReasoningUnit]) -> LoopAnalysis:
    if not units:
        return LoopAnalysis(
            repeated_unit_ratio=0.0,
            adjacent_overlap_ratio=0.0,
            loop_indices=[],
            notes={"total_units": 0},
        )

    normalized = [_normalize_text(unit.text) for unit in units]
    unique_count = len(set(normalized))
    repeated_unit_ratio = 1.0 - (unique_count / len(normalized))

    overlap_scores: List[float] = []
    loop_indices: List[int] = []
    for idx in range(1, len(normalized)):
        prev_tokens = set(normalized[idx - 1].split())
        curr_tokens = set(normalized[idx].split())
        if not prev_tokens or not curr_tokens:
            overlap_scores.append(0.0)
            continue
        jaccard = len(prev_tokens & curr_tokens) / len(prev_tokens | curr_tokens)
        overlap_scores.append(float(jaccard))
        if jaccard >= 0.7:
            loop_indices.append(idx)

    return LoopAnalysis(
        repeated_unit_ratio=float(repeated_unit_ratio),
        adjacent_overlap_ratio=float(np.mean(overlap_scores) if overlap_scores else 0.0),
        loop_indices=loop_indices,
        notes={"total_units": len(units), "unique_units": unique_count},
    )


class ReasoningPipeline:
    def __init__(
        self,
        reasoning_model: ReasoningModel,
        segmenter: BaseSegmenter,
        *,
        cognitive_analyzer: Any | None = None,
        graph_constructor: Any | None = None,
        latent_monitor: Any | None = None,
        pruning_engine: Any | None = None,
        first_step_filter: Any | None = None,
        ccd_logits_processor: Any | None = None,
        enable_self_brake: bool = True,
    ) -> None:
        self.reasoning_model = reasoning_model
        self.segmenter = segmenter
        self.dual_kv_cache = DualKVCache()
        self.cognitive_analyzer = cognitive_analyzer
        self.graph_constructor = graph_constructor
        self.latent_monitor = latent_monitor
        self.pruning_engine = pruning_engine
        self.first_step_filter = first_step_filter
        self.ccd_logits_processor = ccd_logits_processor
        self.enable_self_brake = enable_self_brake

    def run(self, prompt: str, return_hidden_states: bool = False) -> List[AtomicReasoningUnit]:
        generation = self.reasoning_model.generate(
            prompt=prompt,
            return_hidden_states=return_hidden_states,
        )
        chunks = self.segmenter.split(generation.think_text)

        units: List[AtomicReasoningUnit] = []
        for idx, chunk in enumerate(chunks):
            units.append(
                AtomicReasoningUnit(
                    text=chunk,
                    index=idx,
                    cognitive_state=None,
                    label=None,
                    role=None,
                    depth=float(idx / max(1, len(chunks) - 1)),
                    redundancy_score=None,
                    latent_classification=None,
                    metadata={
                        "source": "think_text",
                    },
                )
            )
        return units

    def run_with_details(
        self,
        prompt: str,
        return_hidden_states: bool = False,
    ) -> PipelineRunResult:
        need_hidden = return_hidden_states or (self.latent_monitor is not None)
        generation = self.reasoning_model.generate(
            prompt=prompt,
            return_hidden_states=need_hidden,
            logits_processor=self.ccd_logits_processor,
        )
        chunks = self.segmenter.split(generation.think_text)
        units = [
            AtomicReasoningUnit(
                text=chunk,
                index=i,
                depth=float(i / max(1, len(chunks) - 1)),
                metadata={"source": "think_text"},
            )
            for i, chunk in enumerate(chunks)
        ]

        # Layer 1: Cognitive labeling + transition loops
        transition_self_rate: Optional[float] = None
        transition_matrix: Optional[List[List[float]]] = None
        if self.cognitive_analyzer is not None and units:
            labeled = self.cognitive_analyzer.label_units(units)
            labels = [lu.label for lu in labeled]
            for unit, lu in zip(units, labeled):
                unit.label = lu.label
                unit.cognitive_state = lu.label
                unit.metadata["label_rationale"] = lu.rationale
            try:
                mat = self.cognitive_analyzer.state_transition_matrix(labels)
                transition_matrix = mat.tolist()
                transition_self_rate = float(self.cognitive_analyzer.self_transition_rate(labels))
            except Exception as exc:
                LOGGER.warning("Failed cognitive transition analysis: %s", exc)

        # Layer 2: DAG construction (Progress vs Review roles)
        dag = None
        dag_meta: Dict[str, Any] = {}
        if self.graph_constructor is not None and units:
            try:
                dag = self.graph_constructor.build(units)
                for node in dag.nodes:
                    if 0 <= node.node_id < len(units):
                        units[node.node_id].role = node.role
                        # basic redundancy proxy: review nodes are higher risk
                        units[node.node_id].redundancy_score = 1.0 if node.role == "Review" else 0.0
                dag_meta = {
                    "node_count": len(dag.nodes),
                    "edge_count": len(dag.edges),
                }
            except Exception as exc:
                LOGGER.warning("Failed DAG construction: %s", exc)

        # Layer 3: Latent orbit/fixedpoint detection
        latent = None
        if self.latent_monitor is not None:
            latent = self.latent_monitor.analyze(generation.hidden_state_series)
            if latent is not None:
                for unit in units:
                    unit.latent_classification = latent.classification

        # First-step filter (early discard signal)
        first_step_score: Optional[float] = None
        first_step_rationale: Optional[str] = None
        if self.first_step_filter is not None and units:
            try:
                first_step_score, first_step_rationale = self.first_step_filter.score_first_step(units[0].text)
            except Exception as exc:
                LOGGER.warning("First-step scoring failed: %s", exc)

        loop_analysis = analyze_reasoning_loops(units)

        # Active mitigation: pruning (structural) + self-braking epiphany
        pruning_meta: Dict[str, Any] = {}
        if self.pruning_engine is not None and dag is not None:
            try:
                pruned_dag, pruning_meta = self.pruning_engine.prune(dag)
                units = self.pruning_engine.prune_units_from_dag(units, pruned_dag)  # type: ignore[assignment]
                dag_meta.update({"pruned_node_count": len(pruned_dag.nodes), "pruned_edge_count": len(pruned_dag.edges)})
            except Exception as exc:
                LOGGER.warning("Pruning failed: %s", exc)

        loop_detected = (
            loop_analysis.repeated_unit_ratio >= 0.35
            or loop_analysis.adjacent_overlap_ratio >= 0.5
            or (transition_self_rate is not None and transition_self_rate >= 0.6)
            or (latent is not None and latent.classification in {"FIXEDPOINT", "ORBIT"})
        )
        if self.enable_self_brake and loop_detected:
            try:
                from mitigation import epiphany_sentence

                units.append(
                    AtomicReasoningUnit(
                        text=epiphany_sentence(),
                        index=len(units),
                        label="Interpretation",
                        cognitive_state="Interpretation",
                        role="Progress",
                        depth=1.0,
                        redundancy_score=0.0,
                        latent_classification=(latent.classification if latent is not None else None),
                        metadata={"injected": True, "reason": "loop_detected"},
                    )
                )
            except Exception:
                pass

        return PipelineRunResult(
            units=units,
            full_text=generation.full_text,
            think_text=generation.think_text,
            hidden_states=generation.hidden_states,
            dual_kv_cache=self.dual_kv_cache,
            metadata={
                **(generation.metadata or {}),
                "loop_analysis": {
                    "repeated_unit_ratio": loop_analysis.repeated_unit_ratio,
                    "adjacent_overlap_ratio": loop_analysis.adjacent_overlap_ratio,
                    "loop_indices": loop_analysis.loop_indices,
                    "notes": loop_analysis.notes,
                },
                "cognitive_flow": {
                    "self_transition_rate": transition_self_rate,
                    "transition_matrix": transition_matrix,
                },
                "dag": {
                    **dag_meta,
                    "pruning": pruning_meta,
                },
                "latent": None
                if latent is None
                else {
                    "classification": latent.classification,
                    "mean_successive_cosine": latent.mean_successive_cosine,
                    "min_successive_cosine": latent.min_successive_cosine,
                    "notes": latent.notes,
                },
                "first_step_filter": {
                    "score": first_step_score,
                    "rationale": first_step_rationale,
                },
                "loop_detected": loop_detected,
            },
        )
