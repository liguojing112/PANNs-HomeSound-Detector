"""High-level inference pipeline built on top of PANNs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from audio_processor import ProcessedAudio, load_audio
from config import DEFAULT_TOP_K
from model_loader import LoadedModel, load_panns_model


@dataclass
class Prediction:
    """One label-score prediction pair."""

    label: str
    score: float


@dataclass
class InferenceResult:
    """Structured inference output."""

    predictions: List[Prediction]
    elapsed_ms: float
    duration_sec: float
    used_windows: int
    aggregated_scores: np.ndarray


class PannsPredictor:
    """Convenience wrapper for loading a model once and reusing it."""

    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[str] = None) -> None:
        self.loaded: LoadedModel = load_panns_model(checkpoint_path=checkpoint_path, device=device)

    @property
    def device(self) -> str:
        return self.loaded.device

    @property
    def labels(self) -> Sequence[str]:
        return self.loaded.labels

    def predict_file(self, audio_path: str, top_k: int = DEFAULT_TOP_K) -> InferenceResult:
        processed = load_audio(audio_path)
        return self.predict_processed(processed, top_k=top_k)

    def predict_processed(self, processed: ProcessedAudio, top_k: int = DEFAULT_TOP_K) -> InferenceResult:
        logging.info(
            "Running inference on %.2fs audio at %d Hz using %d window(s)",
            processed.duration_sec,
            processed.sample_rate,
            len(processed.windows),
        )

        import time

        start = time.perf_counter()
        window_scores = []
        for window in processed.windows:
            batch = window.audio[None, :]
            clipwise_output, _ = self.loaded.model.inference(batch)
            window_scores.append(np.asarray(clipwise_output[0], dtype=np.float32))

        aggregated = np.mean(window_scores, axis=0)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        top_indices = np.argsort(aggregated)[::-1][:top_k]
        predictions = [
            Prediction(label=self.labels[index], score=float(aggregated[index]))
            for index in top_indices
        ]
        return InferenceResult(
            predictions=predictions,
            elapsed_ms=elapsed_ms,
            duration_sec=processed.duration_sec,
            used_windows=len(processed.windows),
            aggregated_scores=aggregated,
        )

    def esc50_scores(self, audio_path: str) -> Dict[str, float]:
        """Return raw 527-way scores for downstream ESC-50 zero-shot mapping."""
        result = self.predict_file(audio_path, top_k=DEFAULT_TOP_K)
        return {label: float(result.aggregated_scores[idx]) for idx, label in enumerate(self.labels)}
