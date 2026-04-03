"""Project-wide constants and defaults."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
HOME_PANNS_DIR = Path.home() / "panns_data"
DEFAULT_SAMPLE_RATE = 32000
MIN_AUDIO_SECONDS = 0.5
MAX_DIRECT_AUDIO_SECONDS = 10.0
WINDOW_SECONDS = 5.0
WINDOW_HOP_SECONDS = 2.5
DEFAULT_TOP_K = 5
DEFAULT_ESC50_REPO_ZIP = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
DEFAULT_LABELS_CSV_URL = (
    "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
)
DEFAULT_LABELS_CACHE = PROJECT_ROOT / "checkpoints" / "class_labels_indices.csv"
DEFAULT_PANNS_CHECKPOINT_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
DEFAULT_PANNS_CHECKPOINT_NAME = "Cnn14_mAP=0.431.pth"
