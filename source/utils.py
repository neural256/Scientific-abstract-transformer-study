"""Shared utilities for the abstract classification project.

The scripts in this repository are intended to be run from the project root:

    python source/train_baseline.py

For that reason the path constants below are relative paths rather than
machine-specific absolute paths.
"""

from __future__ import annotations

import ast
import json
import os
import random
from pathlib import Path
from typing import Any

# Keep Matplotlib's font cache inside the workspace. This avoids Windows
# permission errors when the user profile cache directory is not writable.
MPL_CONFIG_DIR = Path(".cache/matplotlib")
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
SAVED_MODELS_DIR = Path("saved_models")

LABEL2ID = {
    "Computer Science": 0,
    "Mathematics": 1,
    "Physics": 2,
    "Biology": 3,
    "Economics": 4,
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}
CLASS_NAMES = [ID2LABEL[i] for i in range(len(ID2LABEL))]

ARCHIVE_TO_FIELD = {
    "Computer Science Archive": "Computer Science",
    "Mathematics Archive": "Mathematics",
    "Physics Archive": "Physics",
    "Quantitative Biology Archive": "Biology",
    "Economics Archive": "Economics",
}


def ensure_project_dirs() -> None:
    """Create output directories used by the training scripts."""
    for path in [DATA_DIR, RESULTS_DIR, FIGURES_DIR, SAVED_MODELS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42, include_torch: bool = True) -> None:
    """Set random seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if not include_torch:
        return

    try:
        import torch
    except Exception:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Select CUDA when available, otherwise CPU."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _coerce_categories(categories: Any) -> list[str]:
    """Normalize raw dataset categories into a list of strings."""
    if categories is None:
        return []
    if isinstance(categories, list):
        return categories
    if isinstance(categories, str):
        stripped = categories.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except (ValueError, SyntaxError):
                pass
        return [stripped]
    return [str(categories)]


def map_arxiv_category_to_field(categories: Any) -> str | None:
    """Convert an arXiv category path into one of the five broad labels.

    Examples:
        ['Computer Science Archive->cs.CV'] -> Computer Science
        ['Physics Archive->astro-ph->astro-ph.EP'] -> Physics
    """
    normalized = _coerce_categories(categories)
    if not normalized:
        return None

    first_category = normalized[0]
    archive_name = first_category.split("->")[0].strip()
    return ARCHIVE_TO_FIELD.get(archive_name)


def clean_text(title: str | None, abstract: str | None) -> str:
    """Combine title and abstract into the single model input text."""
    title = title if title is not None else ""
    abstract = abstract if abstract is not None else ""
    return f"{title.strip()} [SEP] {abstract.strip()}".strip()


def sample_dataframe(
    df: pd.DataFrame,
    max_samples: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a deterministic sample when max_samples is positive."""
    if max_samples is None or max_samples <= 0 or len(df) <= max_samples:
        return df.reset_index(drop=True)

    if "label" in df.columns and max_samples >= df["label"].nunique():
        groups = [(label, group) for label, group in df.groupby("label")]
        allocations: dict[Any, int] = {}
        for label, group in groups:
            proportional = round(max_samples * len(group) / len(df))
            allocations[label] = min(len(group), max(1, int(proportional)))

        while sum(allocations.values()) > max_samples:
            label_to_reduce = max(allocations, key=allocations.get)
            if allocations[label_to_reduce] <= 1:
                break
            allocations[label_to_reduce] -= 1

        while sum(allocations.values()) < max_samples:
            candidates = [
                (label, len(group))
                for label, group in groups
                if allocations[label] < len(group)
            ]
            if not candidates:
                break
            label_to_add = max(candidates, key=lambda item: item[1] - allocations[item[0]])[0]
            allocations[label_to_add] += 1

        sampled_parts = []
        for offset, (label, group) in enumerate(groups):
            sampled_parts.append(
                group.sample(n=allocations[label], random_state=seed + offset)
            )
        return (
            pd.concat(sampled_parts, axis=0)
            .sample(frac=1.0, random_state=seed)
            .reset_index(drop=True)
        )

    return df.sample(n=max_samples, random_state=seed).reset_index(drop=True)


def load_split(
    split_name: str,
    data_dir: Path = DATA_DIR,
    max_samples: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Load one processed split from CSV."""
    path = data_dir / f"{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python source/prepare_dataset.py"
        )

    df = pd.read_csv(path)
    required_columns = {"text", "label", "label_name"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].astype(int)
    return sample_dataframe(df, max_samples=max_samples, seed=seed)


def load_processed_splits(
    data_dir: Path = DATA_DIR,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test CSVs."""
    train_df = load_split("train", data_dir, max_train_samples, seed)
    val_df = load_split("validation", data_dir, max_val_samples, seed)
    test_df = load_split("test", data_dir, max_test_samples, seed)
    return train_df, val_df, test_df


def build_class_level_error_analysis(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    label_names: list[str] = CLASS_NAMES,
) -> dict[str, dict[str, Any]]:
    """Summarize per-class misses and the most common confusion."""
    labels = list(range(len(label_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    analysis: dict[str, dict[str, Any]] = {}

    for idx, label_name in enumerate(label_names):
        row = cm[idx]
        support = int(row.sum())
        correct = int(row[idx])
        errors = int(support - correct)
        error_rate = float(errors / support) if support else 0.0

        off_diagonal = row.copy()
        off_diagonal[idx] = 0
        confused_idx = int(off_diagonal.argmax()) if errors else -1
        confused_label = label_names[confused_idx] if confused_idx >= 0 else None
        confused_count = int(off_diagonal[confused_idx]) if confused_idx >= 0 else 0

        analysis[label_name] = {
            "support": support,
            "correct": correct,
            "errors": errors,
            "error_rate": error_rate,
            "most_confused_with": confused_label,
            "most_confused_count": confused_count,
        }

    return analysis


def compute_classification_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    label_names: list[str] = CLASS_NAMES,
) -> dict[str, Any]:
    """Compute standard and portfolio-friendly classification metrics."""
    labels = list(range(len(label_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "classification_report": report,
        "class_level_error_analysis": build_class_level_error_analysis(
            y_true, y_pred, label_names
        ),
    }


def _json_ready(value: Any) -> Any:
    """Convert NumPy and PyTorch scalar values into JSON-friendly values."""
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if (
        value.__class__.__module__.startswith("torch")
        and hasattr(value, "detach")
        and hasattr(value, "cpu")
    ):
        return value.detach().cpu().tolist()
    return value


def save_json(data: dict[str, Any], output_path: Path) -> None:
    """Save a dictionary as nicely formatted JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(_json_ready(data), file, indent=2)


def plot_confusion_matrix(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    output_path: Path,
    title: str,
    label_names: list[str] = CLASS_NAMES,
) -> None:
    """Save a confusion matrix heatmap as a PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        cbar=False,
    )
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_ablation_results(
    csv_path: Path,
    output_path: Path,
    metric_column: str = "validation_weighted_f1",
) -> None:
    """Plot ablation variants sorted by the selected metric."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    if metric_column not in df.columns:
        raise ValueError(f"{csv_path} does not contain column {metric_column}")

    df = df.sort_values(metric_column, ascending=True)
    plt.figure(figsize=(10, max(5, 0.45 * len(df))))
    plt.barh(df["variant"], df[metric_column], color="#2f6f9f")
    plt.xlabel(metric_column.replace("_", " ").title())
    plt.ylabel("Ablation variant")
    plt.title("Custom Transformer Ablation Study")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
