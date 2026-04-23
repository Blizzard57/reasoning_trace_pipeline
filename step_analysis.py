"""
step_analysis.py
================
Analyse reasoning-trace steps for informativeness and redundancy.

Key concepts
------------
Step length as signal
  Non-informative steps are characteristically *short* — they recapitulate
  prior content ("OK so...", "As computed above…") without advancing the
  solution.  Informative steps introduce new variables, perform non-trivial
  algebra, or draw conclusions that depend on earlier sub-results, and are
  correspondingly *long*.

  Length is measured three ways:
    char_len   – raw character count
    token_len  – whitespace-split word/token count
    sent_len   – number of sentences

  A step is INFORMATIVE  if its length score is >= informative_threshold
  A step is UNINFORMATIVE if its length score is <= uninformative_threshold
  Everything else is MARGINAL.

Redundancy detection
  Two steps are considered redundant if ANY of:
    (a) Jaccard similarity on unigram bags-of-words >= jaccard_thresh
    (b) Normalised edit-distance <= edit_dist_thresh
    (c) Both reference ≥ overlap_math_thresh fraction of the same
        math expressions (LaTeX tokens)
    (d) Cosine similarity on raw TF-IDF vectors >= cosine_thresh

  Redundancy is computed pairwise over ALL step pairs in a trace,
  producing a redundancy matrix and a per-step aggregated score.
"""
from __future__ import annotations
import math, re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ── helpers ──────────────────────────────────────────────────────────────────

_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_MATH_RE = re.compile(r"\\[a-zA-Z]+\{[^}]*\}|\\[a-zA-Z]+|\$[^$]+\$")
_WORD_RE = re.compile(r"\b\w+\b")


def _tokens(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def _math_tokens(text: str) -> List[str]:
    return _MATH_RE.findall(text)


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_RE.split(text.strip()) if s.strip()]

# ── length scoring ───────────────────────────────────────────────────────────

@dataclass
class StepLengthConfig:
    # Thresholds for token count (whitespace-split words)
    # Values tuned on AIME24 traces; adjust via CLI / constructor.
    uninformative_token_max: int = 40    # <= this → definitely short
    informative_token_min:   int = 120   # >= this → definitely informative
    # Weight blend for the composite score (must sum to 1)
    char_weight:  float = 0.3
    token_weight: float = 0.5
    sent_weight:  float = 0.2


@dataclass
class StepLengthScore:
    char_len:    int
    token_len:   int
    sent_len:    int
    norm_score:  float          # 0-1, higher = longer / more informative
    label:       str            # "informative" | "marginal" | "uninformative"


def score_step_length(text: str, cfg: Optional[StepLengthConfig] = None) -> StepLengthScore:
    """
    Compute a length-based informativeness score for a single step.

    We normalise each dimension with a logistic squeeze so that extremely
    long steps don't dominate and the score remains in [0, 1].
    """
    cfg = cfg or StepLengthConfig()
    char_len  = len(text)
    token_len = len(_tokens(text))
    sent_len  = len(_sentences(text))

    # Logistic normalisation: maps 0 → 0, midpoint → 0.5, large → ~1
    def _logistic(x: float, midpoint: float, k: float = 0.02) -> float:
        return 1.0 / (1.0 + math.exp(-k * (x - midpoint)))

    norm_char  = _logistic(char_len,  midpoint=500,  k=0.005)
    norm_tok   = _logistic(token_len, midpoint=80,   k=0.04)
    norm_sent  = _logistic(sent_len,  midpoint=4,    k=0.6)

    score = (cfg.char_weight  * norm_char +
             cfg.token_weight * norm_tok  +
             cfg.sent_weight  * norm_sent)

    # Hard-threshold label
    if token_len <= cfg.uninformative_token_max:
        label = "uninformative"
    elif token_len >= cfg.informative_token_min:
        label = "informative"
    else:
        # Use soft score to break ties in the marginal band
        label = "informative" if score >= 0.55 else "uninformative" if score <= 0.35 else "marginal"

    return StepLengthScore(char_len=char_len, token_len=token_len,
                           sent_len=sent_len, norm_score=round(score, 4),
                           label=label)

# ── pairwise redundancy ──────────────────────────────────────────────────────

@dataclass
class RedundancyConfig:
    jaccard_thresh:      float = 0.55   # unigram bag-of-words overlap
    edit_dist_thresh:    float = 0.30   # 1 - normalised edit distance
    cosine_thresh:       float = 0.70   # TF-IDF cosine sim
    math_overlap_thresh: float = 0.60   # fraction of LaTeX tokens shared


@dataclass
class StepPairRedundancy:
    i:       int
    j:       int
    jaccard: float
    edit:    float          # normalised edit similarity (not distance)
    cosine:  float
    math:    float
    is_redundant: bool
    reason:  str            # which signals fired


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb: return 0.0
    return len(sa & sb) / len(sa | sb)


def _edit_similarity(s1: str, s2: str) -> float:
    """1 - (edit_distance / max_len). O(n^2) — only call on short texts."""
    if not s1 or not s2: return 0.0
    n, m = len(s1), len(s2)
    if n > 1000 or m > 1000:
        # Skip for very long steps; fall back to Jaccard proxy
        return _jaccard(_tokens(s1), _tokens(s2))
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev[j - 1] + cost)
    return 1.0 - (dp[m] / max(n, m))


def _tfidf_cosine(tokens_a: List[str], tokens_b: List[str]) -> float:
    from collections import Counter
    ca, cb = Counter(tokens_a), Counter(tokens_b)
    vocab = set(ca) | set(cb)
    if not vocab: return 0.0
    total = len(tokens_a) + len(tokens_b)
    # IDF proxy: penalise extremely common tokens across a+b
    def tfidf(tok, counter, n):
        tf = counter[tok] / max(n, 1)
        df = (1 if tok in ca else 0) + (1 if tok in cb else 0)
        idf = math.log(3.0 / (1 + df))   # 3 = 1 + num_docs(2)
        return tf * idf
    dot = sum(tfidf(t, ca, len(tokens_a)) * tfidf(t, cb, len(tokens_b)) for t in vocab)
    na = math.sqrt(sum(tfidf(t, ca, len(tokens_a)) ** 2 for t in ca))
    nb = math.sqrt(sum(tfidf(t, cb, len(tokens_b)) ** 2 for t in cb))
    if na * nb == 0: return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))

def _math_overlap(a: str, b: str) -> float:
    ma, mb = set(_math_tokens(a)), set(_math_tokens(b))
    if not ma or not mb: return 0.0
    return len(ma & mb) / max(len(ma), len(mb))


def pairwise_redundancy(step_a: str, step_b: str,
                        cfg: Optional[RedundancyConfig] = None
                        ) -> StepPairRedundancy:
    """Return a StepPairRedundancy for any ordered pair (step_a, step_b)."""
    cfg = cfg or RedundancyConfig()
    idx = (0, 0)   # caller sets i,j
    ta, tb = _tokens(step_a), _tokens(step_b)
    j   = _jaccard(ta, tb)
    ed  = _edit_similarity(step_a, step_b)
    cos = _tfidf_cosine(ta, tb)
    mo  = _math_overlap(step_a, step_b)

    reasons = []
    if j   >= cfg.jaccard_thresh:      reasons.append(f"jaccard={j:.2f}")
    if ed  >= cfg.edit_dist_thresh:    reasons.append(f"edit_sim={ed:.2f}")
    if cos >= cfg.cosine_thresh:       reasons.append(f"cosine={cos:.2f}")
    if mo  >= cfg.math_overlap_thresh: reasons.append(f"math_overlap={mo:.2f}")
    is_redundant = len(reasons) > 0

    return StepPairRedundancy(i=0, j=0, jaccard=round(j, 4),
                              edit=round(ed, 4), cosine=round(cos, 4),
                              math=round(mo, 4), is_redundant=is_redundant,
                              reason="; ".join(reasons) or "none")


# ── trace-level analysis ─────────────────────────────────────────────────────

@dataclass
class TraceAnalysis:
    step_lengths:      List[StepLengthScore]
    redundancy_matrix: List[List[StepPairRedundancy]]  # upper-tri only
    # Per-step aggregate redundancy score (0-1)
    per_step_redundancy: List[float]
    informative_idx:    List[int]
    uninformative_idx:  List[int]
    redundant_pairs:    List[Tuple[int, int, str]]   # (i, j, reason)
    summary:            Dict


def analyse_trace(steps: List[str],
                  length_cfg: Optional[StepLengthConfig] = None,
                  redund_cfg: Optional[RedundancyConfig] = None
                  ) -> TraceAnalysis:
    length_cfg = length_cfg or StepLengthConfig()
    redund_cfg = redund_cfg or RedundancyConfig()

    n = len(steps)
    lengths = [score_step_length(s, length_cfg) for s in steps]
    informative_idx   = [i for i, ls in enumerate(lengths) if ls.label == "informative"]
    uninformative_idx = [i for i, ls in enumerate(lengths) if ls.label == "uninformative"]

    # Build upper-triangular redundancy matrix
    matrix: List[List[StepPairRedundancy]] = [[None]*n for _ in range(n)]  # type:ignore
    redundant_pairs: List[Tuple[int, int, str]] = []
    per_step_scores = [0.0] * n

    for i in range(n):
        for j in range(i + 1, n):
            pr = pairwise_redundancy(steps[i], steps[j], redund_cfg)
            pr.i, pr.j = i, j
            matrix[i][j] = pr
            if pr.is_redundant:
                redundant_pairs.append((i, j, pr.reason))
                per_step_scores[i] += 1
                per_step_scores[j] += 1

    # Normalise per-step redundancy count to [0,1]
    max_possible = max(1, n - 1)
    per_step_red = [min(1.0, s / max_possible) for s in per_step_scores]

    summary = {
        "n_steps": n,
        "n_informative": len(informative_idx),
        "n_uninformative": len(uninformative_idx),
        "n_marginal": n - len(informative_idx) - len(uninformative_idx),
        "n_redundant_pairs": len(redundant_pairs),
        "avg_redundancy_score": round(sum(per_step_red)/max(n,1), 4),
        "avg_token_len": round(sum(ls.token_len for ls in lengths)/max(n,1), 1),
        "informativeness_ratio": round(len(informative_idx)/max(n,1), 3),
    }
    return TraceAnalysis(
        step_lengths=lengths, redundancy_matrix=matrix,
        per_step_redundancy=per_step_red,
        informative_idx=informative_idx,
        uninformative_idx=uninformative_idx,
        redundant_pairs=redundant_pairs, summary=summary)
