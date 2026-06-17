"""Reusable fine-tuning loop for Hugging Face encoder classifiers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from utils import (
    CLASS_NAMES,
    LABEL2ID,
    compute_classification_metrics,
    get_device,
    plot_confusion_matrix,
    save_json,
)


class TextClassificationDataset(Dataset):
    """Simple dataset that keeps raw text and labels until batch collation."""

    def __init__(self, texts: list[str], labels: list[int]):
        self.texts = [str(text) for text in texts]
        self.labels = [int(label) for label in labels]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"text": self.texts[index], "label": self.labels[index]}


def build_collate_fn(tokenizer, max_length: int):
    """Tokenize a batch dynamically to avoid storing all encodings in memory."""

    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = [item["text"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        encoded = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = labels
        return encoded

    return collate_fn


@torch.no_grad()
def predict_hf_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """Return labels and predictions for a Hugging Face classifier."""
    model.eval()
    all_labels: list[int] = []
    all_predictions: list[int] = []

    for batch in data_loader:
        labels = batch.pop("labels")
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(**batch).logits
        predictions = logits.argmax(dim=-1).cpu().tolist()
        all_predictions.extend(predictions)
        all_labels.extend(labels.tolist())

    return all_labels, all_predictions


def train_hf_classifier(
    *,
    model_name: str,
    train_df,
    val_df,
    test_df,
    output_dir: Path,
    metrics_path: Path,
    figure_path: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    warmup_ratio: float,
    weight_decay: float,
    gradient_clip: float,
    display_name: str,
    sample_config: dict[str, int],
) -> dict[str, Any]:
    """Fine-tune a Hugging Face sequence classifier and save artifacts."""
    device = get_device()
    print(f"Using device: {device}")
    print(f"Loading tokenizer/model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(CLASS_NAMES),
        label2id=LABEL2ID,
        id2label={index: label for index, label in enumerate(CLASS_NAMES)},
    )
    model.to(device)

    collate_fn = build_collate_fn(tokenizer, max_length)
    train_loader = DataLoader(
        TextClassificationDataset(train_df["text"].tolist(), train_df["label"].tolist()),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        TextClassificationDataset(val_df["text"].tolist(), val_df["label"].tolist()),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        TextClassificationDataset(test_df["text"].tolist(), test_df["label"].tolist()),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    total_steps = max(1, len(train_loader) * epochs)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    history = []
    started_at = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for batch in progress:
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        val_labels, val_predictions = predict_hf_model(model, val_loader, device)
        val_metrics = compute_classification_metrics(val_labels, val_predictions)
        average_loss = total_loss / max(1, len(train_loader))
        history.append(
            {
                "epoch": epoch,
                "train_loss": average_loss,
                "validation_accuracy": val_metrics["accuracy"],
                "validation_weighted_f1": val_metrics["weighted_f1"],
                "validation_macro_f1": val_metrics["macro_f1"],
            }
        )
        print(
            f"Epoch {epoch}: loss={average_loss:.4f}, "
            f"val_weighted_f1={val_metrics['weighted_f1']:.4f}"
        )

    val_labels, val_predictions = predict_hf_model(model, val_loader, device)
    test_labels, test_predictions = predict_hf_model(model, test_loader, device)

    metrics = {
        "model": display_name,
        "model_name": model_name,
        "class_names": CLASS_NAMES,
        "config": {
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_length": max_length,
            "warmup_ratio": warmup_ratio,
            "weight_decay": weight_decay,
            "gradient_clip": gradient_clip,
            "train_size": len(train_df),
            "validation_size": len(val_df),
            "test_size": len(test_df),
            **sample_config,
        },
        "history": history,
        "training_seconds": time.time() - started_at,
        "validation": compute_classification_metrics(val_labels, val_predictions),
        "test": compute_classification_metrics(test_labels, test_predictions),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_json({"label2id": LABEL2ID, "id2label": metrics["class_names"]}, output_dir / "label_mapping.json")
    save_json(metrics, metrics_path)
    plot_confusion_matrix(
        test_labels,
        test_predictions,
        figure_path,
        title=display_name,
    )

    print(f"Saved model and tokenizer to {output_dir}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved confusion matrix to {figure_path}")
    print(f"Test weighted F1: {metrics['test']['weighted_f1']:.4f}")
    return metrics
