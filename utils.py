"""Shared utilities for logging, timing, downloads, and filesystem helpers."""

from __future__ import annotations

import csv
import logging
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List


def setup_logging(level: str = "INFO") -> None:
    """Configure a consistent project logger."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def time_block() -> Iterator[callable]:
    """Measure elapsed wall-clock time in milliseconds."""
    start = time.perf_counter()
    result = {"elapsed_ms": 0.0}

    def elapsed_ms() -> float:
        return result["elapsed_ms"]

    try:
        yield elapsed_ms
    finally:
        result["elapsed_ms"] = (time.perf_counter() - start) * 1000.0


def download_file(url: str, destination: Path, retries: int = 3) -> Path:
    """Download a file to the target path with basic retry handling."""
    ensure_dir(destination.parent)
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            logging.info("Downloading %s -> %s (attempt %d/%d)", url, destination, attempt, retries)
            _download_with_best_available_client(url, destination)
            return destination
        except (urllib.error.URLError, urllib.error.ContentTooShortError) as exc:
            last_error = exc
            logging.warning("Download attempt %d failed: %s", attempt, exc)
            if destination.exists():
                destination.unlink()
            time.sleep(min(3 * attempt, 10))
        except Exception as exc:
            last_error = exc
            logging.warning("Download attempt %d failed: %s", attempt, exc)
            if destination.exists():
                destination.unlink()
            time.sleep(min(3 * attempt, 10))

    raise RuntimeError(f"Failed to download {url} after {retries} attempts") from last_error


def _download_with_best_available_client(url: str, destination: Path) -> None:
    try:
        import requests

        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with destination.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        return
    except ImportError:
        urllib.request.urlretrieve(url, destination)


def read_labels_from_csv(csv_path: Path) -> List[str]:
    """Read the display names column from a class label csv file."""
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row["display_name"] for row in reader]
