"""PANNs model loading utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch

from compat import install_numba_stub
from config import (
    DEFAULT_LABELS_CACHE,
    DEFAULT_LABELS_CSV_URL,
    DEFAULT_PANNS_CHECKPOINT_NAME,
    DEFAULT_PANNS_CHECKPOINT_URL,
    HOME_PANNS_DIR,
)
from utils import download_file, ensure_dir, read_labels_from_csv

install_numba_stub()


@dataclass
class LoadedModel:
    """Container for the instantiated tagger and metadata."""

    model: object
    labels: List[str]
    device: str
    checkpoint_path: Optional[Path]


def _resolve_device(device: Optional[str] = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_labels() -> List[str]:
    _ensure_runtime_assets()
    from panns_inference import labels as panns_labels

    if panns_labels:
        return list(panns_labels)
    return read_labels_from_csv(HOME_PANNS_DIR / "class_labels_indices.csv")


def _ensure_runtime_assets(checkpoint_path: Optional[Path] = None) -> None:
    """Prepare label metadata and the default checkpoint expected by panns-inference."""
    ensure_dir(HOME_PANNS_DIR)

    home_labels = HOME_PANNS_DIR / "class_labels_indices.csv"
    if not home_labels.exists():
        source_csv = DEFAULT_LABELS_CACHE
        if not source_csv.exists():
            download_file(DEFAULT_LABELS_CSV_URL, source_csv)
        ensure_dir(home_labels.parent)
        home_labels.write_bytes(source_csv.read_bytes())

    if checkpoint_path is None:
        default_checkpoint = HOME_PANNS_DIR / DEFAULT_PANNS_CHECKPOINT_NAME
        if not default_checkpoint.exists() or default_checkpoint.stat().st_size < int(3e8):
            download_file(DEFAULT_PANNS_CHECKPOINT_URL, default_checkpoint)


def load_panns_model(checkpoint_path: Optional[str] = None, device: Optional[str] = None) -> LoadedModel:
    """Load a PANNs AudioTagging model for inference."""
    ckpt = Path(checkpoint_path).expanduser().resolve() if checkpoint_path else None
    _ensure_runtime_assets(checkpoint_path=ckpt)
    from panns_inference import AudioTagging

    resolved_device = _resolve_device(device)
    if ckpt and not ckpt.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt}")

    ensure_dir(DEFAULT_LABELS_CACHE.parent)
    logging.info("Loading PANNs Cnn14 model on %s", resolved_device)
    model = AudioTagging(
        checkpoint_path=str(ckpt) if ckpt else None,
        device=resolved_device,
    )
    labels = _load_labels()
    return LoadedModel(model=model, labels=labels, device=resolved_device, checkpoint_path=ckpt)
