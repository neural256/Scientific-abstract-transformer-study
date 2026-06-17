"""Evaluate a saved model on a processed CSV split.

Examples:

    python source/evaluate.py --model-type baseline
    python source/evaluate.py --model-type custom_transformer --split validation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from utils import (
    CLASS_NAMES,
    FIGURES_DIR,
    RESULTS_DIR,
    SAVED_MODELS_DIR,
    compute_classification_metrics,
    get_device,
    load_split,
    plot_confusion_matrix,
    save_json,
    set_seed,
)


DEFAULT_MODEL_DIRS = {
    "baseline": SAVED_MODELS_DIR / "baseline",
    "distilbert": SAVED_MODELS_DIR / "distilbert",
    "scibert": SAVED_MODELS_DIR / "scibert",
    "custom_transformer": SAVED_MODELS_DIR / "custom_transformer",
}

DEFAULT_METRICS_PATHS = {
    "baseline": RESULTS_DIR / "baseline_metrics.json",
    "distilbert": RESULTS_DIR / "distilbert_metrics.json",
    "scibert": RESULTS_DIR / "scibert_metrics.json",
    "custom_transformer": RESULTS_DIR / "custom_transformer_metrics.json",
}

DEFAULT_FIGURE_PATHS = {
    "baseline": FIGURES_DIR / "confusion_matrix_baseline.png",
    "distilbert": FIGURES_DIR / "confusion_matrix_distilbert.png",
    "scibert": FIGURES_DIR / "confusion_matrix_scibert.png",
    "custom_transformer": FIGURES_DIR / "confusion_matrix_custom_transformer.png",
}


def evaluate_baseline(df, model_dir: Path) -> tuple[list[int], list[int]]:
    model_path = model_dir / "baseline.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing {model_path}. Train the baseline first.")
    pipeline = joblib.load(model_path)
    labels = df["label"].tolist()
    predictions = pipeline.predict(df["text"].tolist()).tolist()
    return labels, predictions


def evaluate_hf(df, model_dir: Path, batch_size: int, max_length: int) -> tuple[list[int], list[int]]:
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from hf_training import TextClassificationDataset, build_collate_fn, predict_hf_model

    if not model_dir.exists():
        raise FileNotFoundError(f"Missing {model_dir}. Train this model first.")

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    loader = DataLoader(
        TextClassificationDataset(df["text"].tolist(), df["label"].tolist()),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=build_collate_fn(tokenizer, max_length),
    )
    return predict_hf_model(model, loader, device)


def evaluate_custom(df, model_dir: Path, batch_size: int) -> tuple[list[int], list[int]]:
    from torch.utils.data import DataLoader

    from train_custom_transformer import (
        CustomAbstractDataset,
        load_custom_artifacts,
        predict_custom_model,
    )

    device = get_device()
    model, vocab, model_config = load_custom_artifacts(model_dir, device)
    dataset = CustomAbstractDataset(
        df["text"].tolist(),
        df["label"].tolist(),
        vocab,
        model_config["max_length"],
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return predict_custom_model(model, loader, device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved classifier.")
    parser.add_argument(
        "--model-type",
        choices=["baseline", "distilbert", "scibert", "custom_transformer"],
        default="baseline",
    )
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--metrics-path", type=Path, default=None)
    parser.add_argument("--figure-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    model_dir = args.model_dir or DEFAULT_MODEL_DIRS[args.model_type]
    metrics_path = args.metrics_path or DEFAULT_METRICS_PATHS[args.model_type]
    figure_path = args.figure_path or DEFAULT_FIGURE_PATHS[args.model_type]
    df = load_split(args.split, max_samples=args.max_samples, seed=args.seed)

    if args.model_type == "baseline":
        labels, predictions = evaluate_baseline(df, model_dir)
    elif args.model_type in {"distilbert", "scibert"}:
        labels, predictions = evaluate_hf(
            df,
            model_dir,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
    else:
        labels, predictions = evaluate_custom(df, model_dir, batch_size=args.batch_size)

    metrics = {
        "model_type": args.model_type,
        "class_names": CLASS_NAMES,
        "evaluated_split": args.split,
        "num_examples": len(df),
        "metrics": compute_classification_metrics(labels, predictions),
    }
    save_json(metrics, metrics_path)
    plot_confusion_matrix(
        labels,
        predictions,
        figure_path,
        title=f"{args.model_type} on {args.split}",
    )

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved confusion matrix to {figure_path}")
    print(f"Weighted F1: {metrics['metrics']['weighted_f1']:.4f}")


if __name__ == "__main__":
    main()
