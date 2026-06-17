"""Run ablations for the custom Transformer encoder.

The default configuration is intentionally small because it trains several
models. Increase samples/epochs when you want publication-quality comparison
numbers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from train_custom_transformer import train_custom_transformer_model
from utils import (
    FIGURES_DIR,
    RESULTS_DIR,
    ensure_project_dirs,
    load_split,
    plot_ablation_results,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run custom Transformer ablations.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=2500)
    parser.add_argument("--max-val-samples", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--output-csv", type=Path, default=RESULTS_DIR / "ablation_results.csv")
    parser.add_argument("--figure-path", type=Path, default=FIGURES_DIR / "ablation_plot.png")
    return parser.parse_args()


def build_ablation_variants() -> list[tuple[str, dict]]:
    """Return model config overrides for each ablation variant."""
    return [
        ("base", {}),
        ("no_positional_encoding", {"use_positional_encoding": False}),
        ("heads_1", {"num_heads": 1}),
        ("heads_2", {"num_heads": 2}),
        ("layers_1", {"num_layers": 1}),
        ("layers_3", {"num_layers": 3}),
        ("ff_dim_128", {"feed_forward_dim": 128}),
        ("ff_dim_512", {"feed_forward_dim": 512}),
        ("dropout_0_1", {"dropout": 0.1}),
        ("dropout_0_3", {"dropout": 0.3}),
    ]


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    set_seed(args.seed)

    train_df = load_split("train", max_samples=args.max_train_samples, seed=args.seed)
    val_df = load_split("validation", max_samples=args.max_val_samples, seed=args.seed)

    base_model_config = {
        "max_vocab_size": 20000,
        "min_freq": 2,
        "max_length": 192,
        "embedding_dim": 128,
        "num_heads": 4,
        "num_layers": 2,
        "feed_forward_dim": 256,
        "dropout": 0.2,
        "use_positional_encoding": True,
    }
    training_config = {
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "max_test_samples": 0,
    }

    rows = []
    for index, (variant_name, overrides) in enumerate(build_ablation_variants(), start=1):
        print(f"\nAblation {index}: {variant_name}")
        set_seed(args.seed)
        model_config = {**base_model_config, **overrides}
        metrics = train_custom_transformer_model(
            train_df=train_df,
            val_df=val_df,
            test_df=None,
            model_config=model_config,
            training_config=training_config,
            save_artifacts=False,
            display_name=f"Custom Transformer Ablation: {variant_name}",
        )

        rows.append(
            {
                "variant": variant_name,
                "use_positional_encoding": model_config["use_positional_encoding"],
                "num_heads": model_config["num_heads"],
                "num_layers": model_config["num_layers"],
                "feed_forward_dim": model_config["feed_forward_dim"],
                "dropout": model_config["dropout"],
                "max_length": model_config["max_length"],
                "embedding_dim": model_config["embedding_dim"],
                "validation_accuracy": metrics["validation"]["accuracy"],
                "validation_weighted_f1": metrics["validation"]["weighted_f1"],
                "validation_macro_f1": metrics["validation"]["macro_f1"],
                "training_seconds": metrics["training_seconds"],
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    plot_ablation_results(args.output_csv, args.figure_path)

    print(f"\nSaved ablation results to {args.output_csv}")
    print(f"Saved ablation plot to {args.figure_path}")


if __name__ == "__main__":
    main()
