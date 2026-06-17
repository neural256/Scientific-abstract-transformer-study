"""Prepare the arXiv category dataset for five-field classification.

Run from the project root:

    python source/prepare_dataset.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from utils import LABEL2ID, clean_text, ensure_project_dirs, map_arxiv_category_to_field


def process_split(dataset_split, split_name: str, output_dir: Path) -> None:
    """Map one Hugging Face split into the processed CSV format."""
    rows = []

    for row in dataset_split:
        field = map_arxiv_category_to_field(row["categories"])
        if field is None:
            continue

        rows.append(
            {
                "id": row["id"],
                "text": clean_text(row["title"], row["abstract"]),
                "label": LABEL2ID[field],
                "label_name": field,
                "original_categories": row["categories"],
                "creation_date": row["creation_date"],
            }
        )

    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split_name}.csv"
    df.to_csv(output_path, index=False)

    print(f"{split_name}: saved {len(df):,} rows to {output_path}")
    print(df["label_name"].value_counts().sort_index())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare processed CSV files.")
    parser.add_argument(
        "--dataset-name",
        default="TimSchopf/arxiv_categories",
        help="Hugging Face dataset name.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory where train/validation/test CSVs are saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()

    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name)

    process_split(dataset["train"], "train", args.output_dir)
    process_split(dataset["validation"], "validation", args.output_dir)
    process_split(dataset["test"], "test", args.output_dir)

    print("\nDataset preparation complete.")


if __name__ == "__main__":
    main()
