"""Explore the raw Hugging Face dataset and broad label distribution.

Run from the project root:

    python source/explore_dataset.py
"""

from __future__ import annotations

from collections import Counter

from datasets import load_dataset

from utils import map_arxiv_category_to_field


def main() -> None:
    print("Loading dataset...")
    dataset = load_dataset("TimSchopf/arxiv_categories")

    print("\nDataset loaded successfully.")
    print(dataset)

    print("\nColumns:")
    print(dataset["train"].column_names)

    print("\nOne training example:")
    print(dataset["train"][0])

    print("\nBroad class distribution:")
    for split_name in ["train", "validation", "test"]:
        labels = []
        for row in dataset[split_name]:
            field = map_arxiv_category_to_field(row["categories"])
            if field is not None:
                labels.append(field)

        counter = Counter(labels)
        print(f"\n{split_name.upper()} distribution:")
        for label, count in sorted(counter.items()):
            print(f"{label}: {count:,}")


if __name__ == "__main__":
    main()
