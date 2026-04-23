from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from models import ReasoningModel

LOGGER = logging.getLogger(__name__)


def _extract_answer_text(result: Any) -> str:
    """Return the post-think answer text from a GenerationResult.

    Slices full_text after </think> so we get the structured answer, not the
    chain-of-thought scratchpad.
    """
    full = (result.full_text or "").strip()
    marker = "</think>"
    idx = full.lower().rfind(marker.lower())
    if idx != -1:
        return full[idx + len(marker):].strip()
    return full


class BaseSegmenter(ABC):
    @abstractmethod
    def split(self, text: str) -> List[str]:
        """Split reasoning text into atomic-ish steps."""


class DelimiterSegmenter(BaseSegmenter):
    def __init__(self, delimiter: str = "\n\n") -> None:
        self.delimiter = delimiter

    def split(self, text: str) -> List[str]:
        chunks = [chunk.strip() for chunk in text.split(self.delimiter)]
        return [chunk for chunk in chunks if chunk]


class KeywordSegmenter(BaseSegmenter):
    def __init__(
        self,
        triggers: Sequence[str] | None = None,
        case_sensitive: bool = False,
    ) -> None:
        self.triggers = list(triggers or ["Alternatively", "Wait", "Hmm", "But wait"])
        self.case_sensitive = case_sensitive
        escaped = [re.escape(token) for token in self.triggers]
        flags = 0 if case_sensitive else re.IGNORECASE
        self.pattern = re.compile(rf"(?=({'|'.join(escaped)}))", flags=flags)
        self._trigger_lookup = (
            {token for token in self.triggers}
            if case_sensitive
            else {token.lower() for token in self.triggers}
        )

    def split(self, text: str) -> List[str]:
        pieces = self.pattern.split(text)
        results: List[str] = []
        buffer = ""
        i = 0
        while i < len(pieces):
            current = pieces[i]
            lookup_value = current if self.case_sensitive else current.lower()
            if lookup_value in self._trigger_lookup:
                if buffer.strip():
                    results.append(buffer.strip())
                buffer = current
            else:
                buffer += current
            i += 1

        if buffer.strip():
            results.append(buffer.strip())
        return results


class HybridSegmenter(BaseSegmenter):
    """Runs delimiter split first, then keyword split inside each chunk."""

    def __init__(
        self,
        delimiter_segmenter: BaseSegmenter | None = None,
        keyword_segmenter: BaseSegmenter | None = None,
    ) -> None:
        self.delimiter_segmenter = delimiter_segmenter or DelimiterSegmenter()
        self.keyword_segmenter = keyword_segmenter or KeywordSegmenter()

    def split(self, text: str) -> List[str]:
        first_pass = self.delimiter_segmenter.split(text)
        if not first_pass:
            return []
        merged: List[str] = []
        for chunk in first_pass:
            merged.extend(self.keyword_segmenter.split(chunk))
        return [chunk.strip() for chunk in merged if chunk.strip()]


@dataclass
class JudgeDecision:
    action: str  # Insert | Merge
    rationale: str


class LLMGraphSegmenter(BaseSegmenter):
    """
    Judge-based iterative splitter.
    Candidate chunks are compared against current graph tail and either:
      - Insert: create new reasoning node (progress)
      - Merge: append into prior node (review/rephrasing)
    """

    def __init__(self, judge_model: ReasoningModel, seed_segmenter: BaseSegmenter | None = None) -> None:
        self.judge_model = judge_model
        self.seed_segmenter = seed_segmenter or DelimiterSegmenter()
        self.last_decisions: List[JudgeDecision] = []

    def split(self, text: str) -> List[str]:
        candidates = self.seed_segmenter.split(text)
        self.last_decisions = []
        if not candidates:
            return []

        nodes: List[str] = [candidates[0]]
        for candidate in candidates[1:]:
            decision = self._judge_decision(nodes[-1], candidate)
            self.last_decisions.append(decision)
            if decision.action == "Insert":
                nodes.append(candidate)
            else:
                nodes[-1] = f"{nodes[-1].rstrip()}\n{candidate.lstrip()}"

            LOGGER.info(
                "Judge decision=%s rationale=%s candidate_preview=%s",
                decision.action,
                decision.rationale,
                candidate[:100].replace("\n", " "),
            )
        return nodes

    def _judge_decision(self, previous: str, candidate: str) -> JudgeDecision:
        judge_prompt = (
            "You are a reasoning-step judge.\n"
            "Decide if candidate text is PROGRESS (new idea/calculation) or REVIEW "
            "(restating/checking previous step).\n"
            "Rules:\n"
            "- Reply with ONLY a single line in this exact format: <action>|<rationale>\n"
            "- <action> must be exactly Insert (for PROGRESS) or Merge (for REVIEW).\n"
            "- Do not include any other text, explanation, or thinking in your reply.\n\n"
            "Example output: Insert|Introduces a new computation.\n\n"
            f"PREVIOUS_STEP:\n{previous}\n\n"
            f"CANDIDATE_STEP:\n{candidate}\n"
        )
        result = self.judge_model.generate(judge_prompt, return_hidden_states=False)
        raw = _extract_answer_text(result).strip()
        # Scan lines bottom-up for first with a valid action
        first_line = ""
        for line in reversed(raw.splitlines()):
            line = line.strip()
            if not line:
                continue
            first_line = line
            if "|" in line:
                potential_action = line.split("|", 1)[0].strip().capitalize()
                if potential_action in {"Insert", "Merge"}:
                    break
        if not first_line:
            first_line = "Merge|No output."

        if "|" in first_line:
            action, rationale = first_line.split("|", 1)
        else:
            action, rationale = first_line, "No explicit rationale."

        action_norm = action.strip().capitalize()
        if action_norm not in {"Insert", "Merge"}:
            # Conservative fallback: merge likely review fragments.
            action_norm = "Merge"
            rationale = f"Unparseable judge output; defaulting to Merge. raw={first_line}"
        return JudgeDecision(action=action_norm, rationale=rationale.strip())

    def decision_stats(self) -> Dict[str, int]:
        stats = {"insert": 0, "merge": 0, "total": len(self.last_decisions)}
        for decision in self.last_decisions:
            if decision.action == "Insert":
                stats["insert"] += 1
            else:
                stats["merge"] += 1
        return stats


def build_segmenter(
    name: str,
    *,
    judge_model: ReasoningModel | None = None,
    delimiter: str = "\n\n",
    triggers: Sequence[str] | None = None,
) -> BaseSegmenter:
    normalized = name.strip().lower()
    if normalized == "delimiter":
        return DelimiterSegmenter(delimiter=delimiter)
    if normalized == "keyword":
        return KeywordSegmenter(triggers=triggers)
    if normalized == "hybrid":
        return HybridSegmenter(
            delimiter_segmenter=DelimiterSegmenter(delimiter=delimiter),
            keyword_segmenter=KeywordSegmenter(triggers=triggers),
        )
    if normalized == "graph":
        if judge_model is None:
            raise ValueError("graph segmenter requires judge_model.")
        return LLMGraphSegmenter(
            judge_model=judge_model,
            seed_segmenter=HybridSegmenter(
                delimiter_segmenter=DelimiterSegmenter(delimiter=delimiter),
                keyword_segmenter=KeywordSegmenter(triggers=triggers),
            ),
        )
    options: Dict[str, str] = {
        "delimiter": "double newline split",
        "keyword": "trigger-word split",
        "hybrid": "delimiter + keyword split",
        "graph": "judge-guided insert/merge split",
    }
    raise ValueError(f"Unknown segmenter '{name}'. Available: {', '.join(options.keys())}")
