"""Download and extract the ESC-50 dataset from GitHub."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

from config import DEFAULT_ESC50_REPO_ZIP
from utils import download_file, ensure_dir, setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="Download ESC-50 from GitHub")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to store the extracted ESC-50 folder")
    parser.add_argument("--url", type=str, default=DEFAULT_ESC50_REPO_ZIP, help="Source zip URL")
    args = parser.parse_args()

    setup_logging("INFO")
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)
    archive_path = output_dir / "ESC-50-master.zip"
    extract_dir = output_dir / "ESC-50-master"

    download_file(args.url, archive_path)
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(output_dir)

    github_root = output_dir / "ESC-50-master"
    if not github_root.exists():
        candidates = [item for item in output_dir.iterdir() if item.is_dir() and item.name.startswith("ESC-50-")]
        if candidates:
            shutil.move(str(candidates[0]), str(extract_dir))
    print(f"ESC-50 extracted to {extract_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
