"""
segmentation_ext.py
====================
Extended segmenters that work across model families:
  - DeepSeek-R1 distills   → keyword triggers ("Alternatively", "Wait", etc.)
  - Gemma 4 / generic chat → paragraph + sentence-boundary + entropy segmenters
  - Universal              → TF-IDF novelty segmenter (model-agnostic)

Also exports build_segmenter_for_model() which auto-selects strategy.
"""
from __future__ import annotations
import re, math
from collections import Counter
from typing import List, Optional, Sequence

from segmentation import (
    BaseSegmenter, DelimiterSegmenter, KeywordSegmenter, HybridSegmenter
)


# ── 1. Sentence-boundary segmenter ──────────────────────────────────────────
# Works for any model. Splits at sentences then merges short ones into the
# next sentence until a minimum character threshold is reached.

class SentenceBoundarySegmenter(BaseSegmenter):
    """
    Split by sentences (period/exclamation/question + whitespace),
    then merge until each chunk has at least `min_chars` characters.
    This produces step-length appropriate chunks even when no keyword
    triggers are present.
    """
    SENT_RE = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, min_chars: int = 200) -> None:
        self.min_chars = min_chars

    def split(self, text: str) -> List[str]:
        raw = self.SENT_RE.split(text.strip())
        merged: List[str] = []
        buf = ""
        for sent in raw:
            buf = (buf + " " + sent).strip() if buf else sent.strip()
            if len(buf) >= self.min_chars:
                merged.append(buf)
                buf = ""
        if buf:
            if merged:
                merged[-1] = merged[-1] + " " + buf
            else:
                merged.append(buf)
        return [c for c in merged if c.strip()]


# ── 2. TF-IDF novelty segmenter ─────────────────────────────────────────────
# Model-agnostic. Splits when the running TF-IDF novelty of a paragraph
# exceeds a threshold (i.e., it introduces genuinely new vocabulary).

def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _tfidf_novelty(new_tokens: List[str], seen_counter: Counter) -> float:
    """Fraction of tokens in new_tokens not seen before."""
    if not new_tokens:
        return 0.0
    novel = sum(1 for t in new_tokens if seen_counter[t] == 0)
    return novel / len(new_tokens)


class NoveltySegmenter(BaseSegmenter):
    """
    Splits a reasoning trace into steps by detecting when a paragraph
    introduces enough novel vocabulary (TF-IDF novelty > threshold).

    Works on DeepSeek, Gemma, and any other model without needing
    keyword triggers.
    """

    def __init__(self, novelty_threshold: float = 0.35,
                 seed_segmenter: Optional[BaseSegmenter] = None) -> None:
        self.novelty_threshold = novelty_threshold
        self.seed = seed_segmenter or DelimiterSegmenter(delimiter="\n\n")

    def split(self, text: str) -> List[str]:
        paragraphs = self.seed.split(text)
        if not paragraphs:
            return []

        steps: List[str] = [paragraphs[0]]
        seen: Counter = Counter(_tokenize(paragraphs[0]))

        for para in paragraphs[1:]:
            tokens = _tokenize(para)
            novelty = _tfidf_novelty(tokens, seen)
            if novelty >= self.novelty_threshold:
                steps.append(para)
            else:
                steps[-1] = steps[-1] + "\n\n" + para
            seen.update(tokens)
        return [s.strip() for s in steps if s.strip()]


# ── 3. Transition-phrase segmenter (extended keyword list) ──────────────────
# Extends the existing KeywordSegmenter with a richer set of transition
# phrases observed in Gemma 4 and Qwen distills.

DEEPSEEK_TRIGGERS = [
    "Alternatively", "Wait", "Hmm", "But wait", "Actually",
    "Let me reconsider", "Let me re-examine", "Let me try",
    "On second thought", "I realize", "I notice",
]
GEMMA_TRIGGERS = [
    "Let me think", "Actually,", "However,", "On the other hand",
    "Let's reconsider", "Wait, let me", "Hmm,", "OK so",
    "Let me re-examine", "I think I made an error",
    "Let me verify", "Let me check",
]
ALL_TRIGGERS = list(dict.fromkeys(DEEPSEEK_TRIGGERS + GEMMA_TRIGGERS))


def build_keyword_segmenter(model_name: str) -> KeywordSegmenter:
    """Return a KeywordSegmenter with triggers appropriate for the model family."""
    n = model_name.lower()
    if "gemma" in n:
        return KeywordSegmenter(triggers=GEMMA_TRIGGERS, case_sensitive=False)
    if "deepseek" in n or "qwen" in n:
        return KeywordSegmenter(triggers=DEEPSEEK_TRIGGERS, case_sensitive=False)
    return KeywordSegmenter(triggers=ALL_TRIGGERS, case_sensitive=False)


# ── 4. Auto-selector ─────────────────────────────────────────────────────────

def build_segmenter_for_model(
    model_name: str,
    strategy: str = "auto",
) -> BaseSegmenter:
    """
    Pick or build the best segmenter for a given model family.

    strategy:
      "auto"     - Chooses based on model name heuristics
      "hybrid"   - Delimiter + keyword (good for DeepSeek/Qwen)
      "novelty"  - TF-IDF novelty segmenter (good for Gemma/generic)
      "sentence" - Sentence-boundary (fallback)
    """
    n = model_name.lower()
    if strategy == "novelty":
        return NoveltySegmenter()
    if strategy == "sentence":
        return SentenceBoundarySegmenter()
    if strategy == "hybrid":
        return HybridSegmenter(
            keyword_segmenter=build_keyword_segmenter(model_name))
    # auto:
    if "gemma" in n:
        # Gemma uses paragraph breaks + novelty, no "<think>" tags
        return NoveltySegmenter(
            seed_segmenter=SentenceBoundarySegmenter(min_chars=150))
    if "deepseek" in n or "qwen" in n or "r1" in n:
        return HybridSegmenter(
            keyword_segmenter=build_keyword_segmenter(model_name))
    # Fallback
    return NoveltySegmenter()
