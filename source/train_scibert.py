"""Fine-tune SciBERT for scientific abstract classification.

Fast default run:

    python source/train_scibert.py

For stronger results, increase samples/epochs, for example:

    python source/train_scibert.py --max-train-samples 0 --max-val-samples 0 --max-test-samples 0 --epochs 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hf_training import train_hf_classifier
from utils import (
    FIGURES_DIR,
    RESULTS_DIR,
    SAVED_MODELS_DIR,
    ensure_project_dirs,
    load_processed_splits,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune SciBERT.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", default="allenai/scibert_scivocab_uncased")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--max-train-samples", type=int, default=3000)
    parser.add_argument("--max-val-samples", type=int, default=1000)
    parser.add_argument("--max-test-samples", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=SAVED_MODELS_DIR / "scibert")
    parser.add_argument("--metrics-path", type=Path, default=RESULTS_DIR / "scibert_metrics.json")
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=FIGURES_DIR / "confusion_matrix_scibert.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    set_seed(args.seed)

    train_df, val_df, test_df = load_processed_splits(
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        seed=args.seed,
    )

    train_hf_classifier(
        model_name=args.model_name,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        output_dir=args.output_dir,
        metrics_path=args.metrics_path,
        figure_path=args.figure_path,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        display_name="Fine-tuned SciBERT",
        sample_config={
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
            "max_test_samples": args.max_test_samples,
        },
    )


if __name__ == "__main__":
    main()
