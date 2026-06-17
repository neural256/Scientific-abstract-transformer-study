"""Train a TF-IDF + Logistic Regression baseline.

Run from the project root:

    python source/train_baseline.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from utils import (
    CLASS_NAMES,
    FIGURES_DIR,
    RESULTS_DIR,
    SAVED_MODELS_DIR,
    compute_classification_metrics,
    ensure_project_dirs,
    load_processed_splits,
    plot_confusion_matrix,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TF-IDF + Logistic Regression baseline."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel jobs for one-vs-rest Logistic Regression. Use 1 on Windows for stability.",
    )
    parser.add_argument("--output-dir", type=Path, default=SAVED_MODELS_DIR / "baseline")
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=RESULTS_DIR / "baseline_metrics.json",
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=FIGURES_DIR / "confusion_matrix_baseline.png",
    )
    return parser.parse_args()


def build_pipeline(
    max_features: int,
    ngram_max: int,
    seed: int,
    n_jobs: int,
) -> Pipeline:
    """Create the baseline model pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    max_features=max_features,
                    min_df=2,
                    ngram_range=(1, ngram_max),
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                OneVsRestClassifier(
                    LogisticRegression(
                        C=2.0,
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=seed,
                        solver="liblinear",
                    ),
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    set_seed(args.seed, include_torch=False)

    train_df, val_df, test_df = load_processed_splits(
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        seed=args.seed,
    )

    pipeline = build_pipeline(
        args.max_features,
        args.ngram_max,
        args.seed,
        args.n_jobs,
    )
    print(f"Training baseline on {len(train_df):,} examples...")
    pipeline.fit(train_df["text"].tolist(), train_df["label"].tolist())

    val_predictions = pipeline.predict(val_df["text"].tolist())
    test_predictions = pipeline.predict(test_df["text"].tolist())

    metrics = {
        "model": "TF-IDF + Logistic Regression",
        "class_names": CLASS_NAMES,
        "config": {
            "seed": args.seed,
            "max_features": args.max_features,
            "ngram_range": [1, args.ngram_max],
            "n_jobs": args.n_jobs,
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
            "max_test_samples": args.max_test_samples,
            "train_size": len(train_df),
            "validation_size": len(val_df),
            "test_size": len(test_df),
        },
        "validation": compute_classification_metrics(
            val_df["label"].tolist(),
            val_predictions,
        ),
        "test": compute_classification_metrics(
            test_df["label"].tolist(),
            test_predictions,
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.output_dir / "baseline.joblib")
    save_json(metrics, args.metrics_path)
    plot_confusion_matrix(
        test_df["label"].tolist(),
        test_predictions,
        args.figure_path,
        title="TF-IDF + Logistic Regression",
    )

    print(f"Saved model to {args.output_dir / 'baseline.joblib'}")
    print(f"Saved metrics to {args.metrics_path}")
    print(f"Saved confusion matrix to {args.figure_path}")
    print(f"Test weighted F1: {metrics['test']['weighted_f1']:.4f}")


if __name__ == "__main__":
    main()
