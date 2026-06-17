"""Train the manual PyTorch Transformer encoder classifier.

Fast default run:

    python source/train_custom_transformer.py

For a stronger custom model, increase capacity and train on all data:

    python source/train_custom_transformer.py --max-train-samples 0 --max-val-samples 0 --max-test-samples 0 --epochs 8 --embedding-dim 256 --num-heads 8 --num-layers 4 --feed-forward-dim 512
"""

from __future__ import annotations

import argparse
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from transformer_model import CustomTransformerClassifier
from utils import (
    CLASS_NAMES,
    FIGURES_DIR,
    ID2LABEL,
    LABEL2ID,
    RESULTS_DIR,
    SAVED_MODELS_DIR,
    compute_classification_metrics,
    ensure_project_dirs,
    get_device,
    load_processed_splits,
    plot_confusion_matrix,
    save_json,
    set_seed,
)


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]+|\d+(?:\.\d+)?")


def simple_tokenize(text: str) -> list[str]:
    """Tokenize text with a compact regex suitable for a from-scratch baseline."""
    return TOKEN_PATTERN.findall(str(text).lower())


def build_vocabulary(
    texts: list[str],
    max_vocab_size: int = 30000,
    min_freq: int = 2,
) -> dict[str, int]:
    """Build a word-level vocabulary from training text only."""
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(simple_tokenize(text))

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, count in counter.most_common():
        if count < min_freq:
            continue
        if len(vocab) >= max_vocab_size:
            break
        vocab[token] = len(vocab)
    return vocab


def encode_text(
    text: str,
    vocab: dict[str, int],
    max_length: int,
) -> tuple[list[int], list[int]]:
    """Convert text into token ids plus an attention mask."""
    token_ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in simple_tokenize(text)]
    token_ids = token_ids[:max_length]
    attention_mask = [1] * len(token_ids)

    pad_length = max_length - len(token_ids)
    if pad_length > 0:
        token_ids.extend([vocab[PAD_TOKEN]] * pad_length)
        attention_mask.extend([0] * pad_length)

    return token_ids, attention_mask


class CustomAbstractDataset(Dataset):
    """Pre-encoded dataset for the custom Transformer."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        vocab: dict[str, int],
        max_length: int,
    ):
        self.labels = [int(label) for label in labels]
        self.encoded = [encode_text(text, vocab, max_length) for text in texts]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        input_ids, attention_mask = self.encoded[index]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label": torch.tensor(self.labels[index], dtype=torch.long),
        }


def validate_model_config(model_config: dict[str, Any]) -> None:
    """Fail early for invalid Transformer sizes."""
    embedding_dim = model_config["embedding_dim"]
    num_heads = model_config["num_heads"]
    if embedding_dim % num_heads != 0:
        raise ValueError("embedding_dim must be divisible by num_heads")
    if model_config["max_length"] <= 0:
        raise ValueError("max_length must be positive")


def build_custom_model(
    vocab_size: int,
    model_config: dict[str, Any],
) -> CustomTransformerClassifier:
    """Create a classifier from a config dictionary."""
    validate_model_config(model_config)
    return CustomTransformerClassifier(
        vocab_size=vocab_size,
        num_classes=len(CLASS_NAMES),
        embedding_dim=model_config["embedding_dim"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        feed_forward_dim=model_config["feed_forward_dim"],
        max_length=model_config["max_length"],
        dropout=model_config["dropout"],
        pad_token_id=0,
        use_positional_encoding=model_config["use_positional_encoding"],
    )


@torch.no_grad()
def predict_custom_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """Return labels and predictions for a custom Transformer."""
    model.eval()
    labels: list[int] = []
    predictions: list[int] = []

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        batch_labels = batch["label"].tolist()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        batch_predictions = logits.argmax(dim=-1).cpu().tolist()
        labels.extend(batch_labels)
        predictions.extend(batch_predictions)

    return labels, predictions


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: float,
    epoch: int,
    epochs: int,
) -> float:
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    progress = tqdm(data_loader, desc=f"Epoch {epoch}/{epochs}")

    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(1, len(data_loader))


def save_custom_artifacts(
    model: nn.Module,
    vocab: dict[str, int],
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    output_dir: Path,
) -> None:
    """Save checkpoint and readable config files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocab": vocab,
        "model_config": model_config,
        "training_config": training_config,
        "label2id": LABEL2ID,
        "id2label": ID2LABEL,
    }
    torch.save(checkpoint, output_dir / "custom_transformer.pt")
    save_json(vocab, output_dir / "vocab.json")
    save_json(model_config, output_dir / "model_config.json")
    save_json(training_config, output_dir / "training_config.json")


def _torch_load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """Load checkpoints across PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_custom_artifacts(
    model_dir: Path,
    device: torch.device | None = None,
) -> tuple[CustomTransformerClassifier, dict[str, int], dict[str, Any]]:
    """Load a saved custom Transformer model, vocabulary, and config."""
    device = device or get_device()
    checkpoint_path = model_dir / "custom_transformer.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing {checkpoint_path}. Run: python source/train_custom_transformer.py"
        )

    checkpoint = _torch_load_checkpoint(checkpoint_path, device)
    vocab = checkpoint["vocab"]
    model_config = checkpoint["model_config"]
    model = build_custom_model(len(vocab), model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, vocab, model_config


def train_custom_transformer_model(
    *,
    train_df,
    val_df,
    test_df,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    output_dir: Path | None = None,
    metrics_path: Path | None = None,
    figure_path: Path | None = None,
    save_artifacts: bool = True,
    display_name: str = "Custom Transformer Encoder",
) -> dict[str, Any]:
    """Train and evaluate the custom Transformer."""
    validate_model_config(model_config)
    device = get_device()
    print(f"Using device: {device}")

    vocab = build_vocabulary(
        train_df["text"].tolist(),
        max_vocab_size=model_config["max_vocab_size"],
        min_freq=model_config["min_freq"],
    )
    print(f"Vocabulary size: {len(vocab):,}")

    train_dataset = CustomAbstractDataset(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        vocab,
        model_config["max_length"],
    )
    val_dataset = CustomAbstractDataset(
        val_df["text"].tolist(),
        val_df["label"].tolist(),
        vocab,
        model_config["max_length"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
    )

    test_loader = None
    if test_df is not None:
        test_dataset = CustomAbstractDataset(
            test_df["text"].tolist(),
            test_df["label"].tolist(),
            vocab,
            model_config["max_length"],
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=training_config["batch_size"],
            shuffle=False,
        )

    model = build_custom_model(len(vocab), model_config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )

    history = []
    started_at = time.time()
    for epoch in range(1, training_config["epochs"] + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            training_config["gradient_clip"],
            epoch,
            training_config["epochs"],
        )
        val_labels, val_predictions = predict_custom_model(model, val_loader, device)
        val_metrics = compute_classification_metrics(val_labels, val_predictions)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_accuracy": val_metrics["accuracy"],
                "validation_weighted_f1": val_metrics["weighted_f1"],
                "validation_macro_f1": val_metrics["macro_f1"],
            }
        )
        print(
            f"Epoch {epoch}: loss={train_loss:.4f}, "
            f"val_weighted_f1={val_metrics['weighted_f1']:.4f}"
        )

    val_labels, val_predictions = predict_custom_model(model, val_loader, device)
    metrics = {
        "model": display_name,
        "class_names": CLASS_NAMES,
        "model_config": model_config,
        "training_config": {
            **training_config,
            "train_size": len(train_df),
            "validation_size": len(val_df),
            "test_size": len(test_df) if test_df is not None else 0,
            "vocab_size": len(vocab),
        },
        "history": history,
        "training_seconds": time.time() - started_at,
        "validation": compute_classification_metrics(val_labels, val_predictions),
    }

    test_labels: list[int] | None = None
    test_predictions: list[int] | None = None
    if test_loader is not None:
        test_labels, test_predictions = predict_custom_model(model, test_loader, device)
        metrics["test"] = compute_classification_metrics(test_labels, test_predictions)

    if save_artifacts and output_dir is not None:
        save_custom_artifacts(model, vocab, model_config, metrics["training_config"], output_dir)
    if metrics_path is not None:
        save_json(metrics, metrics_path)
    if figure_path is not None and test_labels is not None and test_predictions is not None:
        plot_confusion_matrix(
            test_labels,
            test_predictions,
            figure_path,
            title=display_name,
        )

    if output_dir is not None:
        print(f"Saved custom model artifacts to {output_dir}")
    if metrics_path is not None:
        print(f"Saved metrics to {metrics_path}")
    if figure_path is not None:
        print(f"Saved confusion matrix to {figure_path}")
    if "test" in metrics:
        print(f"Test weighted F1: {metrics['test']['weighted_f1']:.4f}")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train custom Transformer encoder.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=10000)
    parser.add_argument("--max-val-samples", type=int, default=2000)
    parser.add_argument("--max-test-samples", type=int, default=2000)
    parser.add_argument("--max-vocab-size", type=int, default=30000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--feed-forward-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--no-positional-encoding",
        action="store_true",
        help="Disable positional encoding for ablation/debugging.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SAVED_MODELS_DIR / "custom_transformer",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=RESULTS_DIR / "custom_transformer_metrics.json",
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=FIGURES_DIR / "confusion_matrix_custom_transformer.png",
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

    model_config = {
        "max_vocab_size": args.max_vocab_size,
        "min_freq": args.min_freq,
        "max_length": args.max_length,
        "embedding_dim": args.embedding_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "feed_forward_dim": args.feed_forward_dim,
        "dropout": args.dropout,
        "use_positional_encoding": not args.no_positional_encoding,
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
        "max_test_samples": args.max_test_samples,
    }

    train_custom_transformer_model(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        model_config=model_config,
        training_config=training_config,
        output_dir=args.output_dir,
        metrics_path=args.metrics_path,
        figure_path=args.figure_path,
        save_artifacts=True,
    )


if __name__ == "__main__":
    main()
