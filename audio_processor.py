"""Audio loading, validation, resampling, and segmentation helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from compat import install_numba_stub
from config import (
    DEFAULT_SAMPLE_RATE,
    MAX_DIRECT_AUDIO_SECONDS,
    MIN_AUDIO_SECONDS,
    WINDOW_HOP_SECONDS,
    WINDOW_SECONDS,
)

install_numba_stub()

import librosa


@dataclass
class AudioWindow:
    """Represents one inference chunk from an audio file."""

    audio: np.ndarray
    start_sec: float
    end_sec: float


@dataclass
class ProcessedAudio:
    """Prepared audio and chunk metadata."""

    audio: np.ndarray
    sample_rate: int
    duration_sec: float
    windows: List[AudioWindow]


def load_audio(audio_path: str, sample_rate: int = DEFAULT_SAMPLE_RATE) -> ProcessedAudio:
    """Load an audio file, resample to the target sample rate, and create inference windows."""
    path = Path(audio_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    logging.info("Loading audio from %s", path)
    try:
        audio, _ = librosa.load(path.as_posix(), sr=sample_rate, mono=True)
    except Exception as exc:
        raise ValueError(f"Unsupported or unreadable audio file: {path}") from exc

    duration = len(audio) / sample_rate if len(audio) else 0.0
    if duration < MIN_AUDIO_SECONDS:
        raise ValueError(f"Audio too short: {duration:.2f}s < {MIN_AUDIO_SECONDS:.2f}s")

    windows = create_windows(audio, sample_rate)
    return ProcessedAudio(audio=audio.astype(np.float32), sample_rate=sample_rate, duration_sec=duration, windows=windows)


def create_windows(audio: np.ndarray, sample_rate: int) -> List[AudioWindow]:
    """Split long audio into overlapping windows for stable inference."""
    duration = len(audio) / sample_rate
    if duration <= MAX_DIRECT_AUDIO_SECONDS:
        return [AudioWindow(audio=audio.astype(np.float32), start_sec=0.0, end_sec=duration)]

    window_samples = int(WINDOW_SECONDS * sample_rate)
    hop_samples = int(WINDOW_HOP_SECONDS * sample_rate)
    windows: List[AudioWindow] = []
    last_start = max(len(audio) - window_samples, 0)

    for start in range(0, last_start + 1, hop_samples):
        end = min(start + window_samples, len(audio))
        chunk = audio[start:end]
        if len(chunk) < window_samples:
            chunk = np.pad(chunk, (0, window_samples - len(chunk)))
        windows.append(
            AudioWindow(
                audio=chunk.astype(np.float32),
                start_sec=start / sample_rate,
                end_sec=min(end / sample_rate, duration),
            )
        )

    if not windows:
        windows.append(AudioWindow(audio=audio.astype(np.float32), start_sec=0.0, end_sec=duration))

    logging.info("Audio longer than %.1fs, created %d overlapping windows", MAX_DIRECT_AUDIO_SECONDS, len(windows))
    return windows
