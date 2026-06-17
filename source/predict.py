"""Predict the broad academic field for a title and abstract.

Example:

    python source/predict.py --model-type baseline --title "Graph neural networks" --abstract "We study..."
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from utils import CLASS_NAMES, SAVED_MODELS_DIR, clean_text, get_device


DEFAULT_MODEL_DIRS = {
    "baseline": SAVED_MODELS_DIR / "baseline",
    "distilbert": SAVED_MODELS_DIR / "distilbert",
    "scibert": SAVED_MODELS_DIR / "scibert",
    "custom_transformer": SAVED_MODELS_DIR / "custom_transformer",
}


def resolve_model_dir(model_type: str, model_dir: Path | None) -> Path:
    return model_dir if model_dir is not None else DEFAULT_MODEL_DIRS[model_type]


def predict_with_baseline(text: str, model_dir: Path) -> tuple[str, list[float]]:
    model_path = model_dir / "baseline.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing {model_path}. Train the baseline first.")

    pipeline = joblib.load(model_path)
    raw_probabilities = pipeline.predict_proba([text])[0]
    probabilities = [0.0] * len(CLASS_NAMES)
    class_ids = pipeline.named_steps["classifier"].classes_
    for class_id, probability in zip(class_ids, raw_probabilities):
        probabilities[int(class_id)] = float(probability)
    predicted_id = max(range(len(probabilities)), key=probabilities.__getitem__)
    return CLASS_NAMES[predicted_id], probabilities


def predict_with_hf_model(
    text: str,
    model_dir: Path,
    max_length: int,
) -> tuple[str, list[float]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not model_dir.exists():
        raise FileNotFoundError(f"Missing {model_dir}. Train this model first.")

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    encoded = tokenizer(
        [text],
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits
    probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
    predicted_id = int(torch.tensor(probabilities).argmax().item())
    return CLASS_NAMES[predicted_id], [float(value) for value in probabilities]


def predict_with_custom_transformer(
    text: str,
    model_dir: Path,
) -> tuple[str, list[float]]:
    import torch

    from train_custom_transformer import encode_text, load_custom_artifacts

    device = get_device()
    model, vocab, model_config = load_custom_artifacts(model_dir, device)
    input_ids, attention_mask = encode_text(text, vocab, model_config["max_length"])
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    mask_tensor = torch.tensor([attention_mask], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(input_ids=input_tensor, attention_mask=mask_tensor)
    probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
    predicted_id = int(torch.tensor(probabilities).argmax().item())
    return CLASS_NAMES[predicted_id], [float(value) for value in probabilities]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a field for one abstract.")
    parser.add_argument(
        "--model-type",
        choices=["baseline", "distilbert", "scibert", "custom_transformer"],
        default="baseline",
    )
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--title", required=True)
    parser.add_argument("--abstract", required=True)
    parser.add_argument("--max-length", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = resolve_model_dir(args.model_type, args.model_dir)
    text = clean_text(args.title, args.abstract)

    if args.model_type == "baseline":
        predicted_label, probabilities = predict_with_baseline(text, model_dir)
    elif args.model_type in {"distilbert", "scibert"}:
        predicted_label, probabilities = predict_with_hf_model(
            text,
            model_dir,
            args.max_length,
        )
    else:
        predicted_label, probabilities = predict_with_custom_transformer(text, model_dir)

    print(f"Predicted field: {predicted_label}")
    print("\nClass probabilities:")
    for label, probability in zip(CLASS_NAMES, probabilities):
        print(f"  {label}: {probability:.4f}")


if __name__ == "__main__":
    main()
