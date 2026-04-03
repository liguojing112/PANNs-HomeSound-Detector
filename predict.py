"""Command-line entry point for single-file inference and optional evaluation."""

from __future__ import annotations

import argparse
import logging
import sys
import time

from evaluator import ESC50Evaluator
from inference import PannsPredictor
from utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PANNs home sound detector demo")
    parser.add_argument("--audio", type=str, help="Path to an audio file for single-file prediction")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional local PANNs checkpoint path")
    parser.add_argument("--device", type=str, default=None, help="Force device, e.g. cpu or cuda")
    parser.add_argument("--top_k", type=int, default=5, help="Number of labels to print")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--eval", action="store_true", help="Run ESC-50 evaluation instead of single-file inference")
    parser.add_argument("--data_root", type=str, default="./ESC-50-master", help="ESC-50 dataset root for evaluation")
    parser.add_argument("--test_fold", type=int, default=1, help="ESC-50 test fold")
    parser.add_argument("--subset_ratio", type=float, default=1.0, help="Subset ratio for evaluation")
    parser.add_argument("--confusion_path", type=str, default="confusion_matrix.png", help="Confusion matrix output path")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.log_level)

    if not args.audio and not args.eval:
        parser.error("Provide --audio for inference or --eval for ESC-50 evaluation.")

    try:
        model_start = time.perf_counter()
        predictor = PannsPredictor(checkpoint_path=args.checkpoint, device=args.device)
        model_load_sec = time.perf_counter() - model_start
        print(f"Loaded PANNs model ({predictor.device.upper()}) in {model_load_sec:.2f}s.")

        if args.eval:
            evaluator = ESC50Evaluator(predictor)
            summary = evaluator.evaluate(
                data_root=args.data_root,
                test_fold=args.test_fold,
                subset_ratio=args.subset_ratio,
                output_path=args.confusion_path,
            )
            print(f"Test samples: {summary.sample_count}")
            print(f"Top-1 accuracy: {summary.top1_accuracy:.3f}")
            print(f"Top-5 accuracy: {summary.top5_accuracy:.3f}")
            print(f"Confusion matrix saved to {summary.confusion_matrix_path}")
            return 0

        result = predictor.predict_file(args.audio, top_k=args.top_k)
        print(
            f"Processing audio: {args.audio} "
            f"({result.duration_sec:.2f}s, 32000Hz mono, windows={result.used_windows})"
        )
        print("Top-5 predictions:")
        for idx, prediction in enumerate(result.predictions, start=1):
            print(f"  {idx}. {prediction.label} ({prediction.score:.4f})")
        print(f"Inference time: {result.elapsed_ms:.2f} ms")
        return 0
    except Exception as exc:
        logging.exception("Prediction failed")
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
