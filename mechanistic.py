from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


@dataclass
class LatentAnalysis:
    classification: str  # FIXEDPOINT | ORBIT | SLIDER | UNKNOWN
    mean_successive_cosine: float
    min_successive_cosine: float
    similarity_series: np.ndarray
    notes: Dict[str, Any]


class LatentMonitor:
    """
    Monitors a hidden-state trajectory (typically last-layer, last-token over time)
    to detect stabilization into cyclic behaviors.

    Input convention:
      hidden_state_series: torch.Tensor of shape (steps, hidden)
    """

    def __init__(
        self,
        fixedpoint_threshold: float = 0.999,
        orbit_peak_threshold: float = 0.15,
        use_fft: bool = True,
    ) -> None:
        self.fixedpoint_threshold = fixedpoint_threshold
        self.orbit_peak_threshold = orbit_peak_threshold
        self.use_fft = use_fft

    def analyze(self, hidden_state_series: Optional[torch.Tensor]) -> Optional[LatentAnalysis]:
        if hidden_state_series is None:
            return None
        if hidden_state_series.ndim != 2 or hidden_state_series.shape[0] < 4:
            return LatentAnalysis(
                classification="UNKNOWN",
                mean_successive_cosine=0.0,
                min_successive_cosine=0.0,
                similarity_series=np.array([], dtype=np.float32),
                notes={"reason": "insufficient_steps_or_bad_shape", "shape": tuple(hidden_state_series.shape)},
            )

        series = hidden_state_series.float().cpu()
        sims = self._successive_cosine(series)

        mean_sim = float(np.mean(sims))
        min_sim = float(np.min(sims))

        classification = "UNKNOWN"
        notes: Dict[str, Any] = {}

        # Fixedpoint: extremely high successive similarity for a sustained period.
        if mean_sim >= self.fixedpoint_threshold and min_sim >= (self.fixedpoint_threshold - 0.01):
            classification = "FIXEDPOINT"
            notes["rule"] = "mean/min above fixedpoint thresholds"
            return LatentAnalysis(
                classification=classification,
                mean_successive_cosine=mean_sim,
                min_successive_cosine=min_sim,
                similarity_series=sims,
                notes=notes,
            )

        if self.use_fft and sims.size >= 8:
            peak, peak_bin = self._fft_peak(sims)
            notes["fft_peak"] = float(peak)
            notes["fft_peak_bin"] = int(peak_bin)
            if peak >= self.orbit_peak_threshold:
                classification = "ORBIT"
            else:
                classification = "SLIDER"
        else:
            # Heuristic: low variance in similarity suggests slider-like stabilization.
            std = float(np.std(sims)) if sims.size else 0.0
            notes["std"] = std
            classification = "SLIDER" if std < 0.02 else "UNKNOWN"

        return LatentAnalysis(
            classification=classification,
            mean_successive_cosine=mean_sim,
            min_successive_cosine=min_sim,
            similarity_series=sims,
            notes=notes,
        )

    @staticmethod
    def _successive_cosine(series: torch.Tensor) -> np.ndarray:
        a = series[:-1]
        b = series[1:]
        a = a / (a.norm(dim=1, keepdim=True) + 1e-8)
        b = b / (b.norm(dim=1, keepdim=True) + 1e-8)
        cos = (a * b).sum(dim=1).clamp(-1.0, 1.0)
        return cos.detach().cpu().numpy().astype(np.float32)

    @staticmethod
    def _fft_peak(similarity_series: np.ndarray) -> tuple[float, int]:
        x = similarity_series.astype(np.float32)
        x = x - float(np.mean(x))
        spectrum = np.abs(np.fft.rfft(x))
        if spectrum.size <= 1:
            return 0.0, 0
        spectrum[0] = 0.0  # remove DC
        peak_bin = int(np.argmax(spectrum))
        peak_val = float(spectrum[peak_bin] / (np.sum(spectrum) + 1e-8))
        return peak_val, peak_bin

