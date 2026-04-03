"""Dedicated ESC-50 evaluation entry point."""

from __future__ import annotations

import argparse
import logging
import sys

from evaluator import ESC50Evaluator
from inference import PannsPredictor
from utils import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate zero-shot PANNs on ESC-50")
    parser.add_argument("--data_root", type=str, required=True, help="Path to the ESC-50 dataset root")
    parser.add_argument("--test_fold", type=int, default=1, help="ESC-50 test fold")
    parser.add_argument("--subset_ratio", type=float, default=1.0, help="Subset ratio for quick demos")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional local PANNs checkpoint path")
    parser.add_argument("--device", type=str, default=None, help="Force device, e.g. cpu or cuda")
    parser.add_argument("--output", type=str, default="confusion_matrix.png", help="Confusion matrix output path")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()

    setup_logging(args.log_level)
    try:
        predictor = PannsPredictor(checkpoint_path=args.checkpoint, device=args.device)
        evaluator = ESC50Evaluator(predictor)
        summary = evaluator.evaluate(
            data_root=args.data_root,
            test_fold=args.test_fold,
            subset_ratio=args.subset_ratio,
            output_path=args.output,
        )
        print(f"Test samples: {summary.sample_count}")
        print(f"Top-1 accuracy: {summary.top1_accuracy:.3f}")
        print(f"Top-5 accuracy: {summary.top5_accuracy:.3f}")
        print(f"Confusion matrix saved to {summary.confusion_matrix_path}")
        return 0
    except Exception as exc:
        logging.exception("Evaluation failed")
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
