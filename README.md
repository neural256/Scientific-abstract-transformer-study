# Comparative Study of Transformer Architectures for Scientific Abstract Classification

This project compares classical and Transformer-based NLP models for classifying scientific paper abstracts into five broad academic fields. It is designed as a resume-level deep learning portfolio project with reproducible training scripts, saved metrics, confusion matrices, model artifacts, prediction, evaluation, and an ablation study for a manually implemented PyTorch Transformer encoder.

## Project Goal

Given a paper title and abstract, predict one of:

1. Computer Science
2. Mathematics
3. Physics
4. Biology
5. Economics

The comparison covers:

- TF-IDF + Logistic Regression baseline
- Fine-tuned DistilBERT
- Fine-tuned SciBERT
- Custom Transformer Encoder implemented manually in PyTorch
- Ablation study of custom Transformer components

## Dataset

The project uses the Hugging Face dataset `TimSchopf/arxiv_categories`.

Raw columns:

- `id`
- `title`
- `abstract`
- `categories`
- `creation_date`

Splits:

- `train`
- `validation`
- `test`

Processed files are written to:

- `data/processed/train.csv`
- `data/processed/validation.csv`
- `data/processed/test.csv`

The processed text input is:

```text
title [SEP] abstract
```

## Label Mapping

Labels are mapped from the archive name before `Archive` in the first category path:

| Raw archive prefix | Final label |
| --- | --- |
| `Computer Science Archive` | Computer Science |
| `Mathematics Archive` | Mathematics |
| `Physics Archive` | Physics |
| `Quantitative Biology Archive` | Biology |
| `Economics Archive` | Economics |

The numeric labels are defined in `source/utils.py`.

## Model Architectures

### TF-IDF + Logistic Regression

A scikit-learn pipeline using `TfidfVectorizer` with unigram/bigram features and a balanced Logistic Regression classifier. This is the fast, interpretable baseline.

### DistilBERT

Fine-tunes `distilbert-base-uncased` with a sequence classification head using Hugging Face Transformers and a manual PyTorch training loop.

### SciBERT

Fine-tunes `allenai/scibert_scivocab_uncased`, a BERT model pretrained on scientific text. This is expected to be especially competitive for scientific abstracts.

### Custom Transformer Encoder

Implemented manually in `source/transformer_model.py`, including:

- token embeddings
- sinusoidal positional encoding
- scaled dot-product attention
- multi-head self-attention
- residual connections
- layer normalization
- feed-forward network
- dropout
- masked mean pooling
- classification head

The custom model intentionally does not use `torch.nn.TransformerEncoder`.

## Installation

Use Python 3.11.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

## Prepare Data

Run from the project root:

```bash
python source/prepare_dataset.py
```

Optional dataset inspection:

```bash
python source/explore_dataset.py
```

## Train Models

All scripts are runnable from the project root.

### Baseline

```bash
python source/train_baseline.py
```

Outputs:

- `saved_models/baseline/baseline.joblib`
- `results/baseline_metrics.json`
- `figures/confusion_matrix_baseline.png`

### DistilBERT

Fast laptop-friendly default:

```bash
python source/train_distilbert.py
```

Stronger run:

```bash
python source/train_distilbert.py --max-train-samples 0 --max-val-samples 0 --max-test-samples 0 --epochs 3
```

Outputs:

- `saved_models/distilbert/`
- `results/distilbert_metrics.json`
- `figures/confusion_matrix_distilbert.png`

### SciBERT

Fast laptop-friendly default:

```bash
python source/train_scibert.py
```

Stronger run:

```bash
python source/train_scibert.py --max-train-samples 0 --max-val-samples 0 --max-test-samples 0 --epochs 3
```

Outputs:

- `saved_models/scibert/`
- `results/scibert_metrics.json`
- `figures/confusion_matrix_scibert.png`

### Custom Transformer

Fast laptop-friendly default:

```bash
python source/train_custom_transformer.py
```

Stronger run:

```bash
python source/train_custom_transformer.py --max-train-samples 0 --max-val-samples 0 --max-test-samples 0 --epochs 8 --embedding-dim 256 --num-heads 8 --num-layers 4 --feed-forward-dim 512
```

Outputs:

- `saved_models/custom_transformer/custom_transformer.pt`
- `saved_models/custom_transformer/vocab.json`
- `saved_models/custom_transformer/model_config.json`
- `results/custom_transformer_metrics.json`
- `figures/confusion_matrix_custom_transformer.png`

## Run Ablation Study

```bash
python source/run_ablation.py
```

Outputs:

- `results/ablation_results.csv`
- `figures/ablation_plot.png`

The ablation compares:

- with vs without positional encoding
- different attention head counts
- different encoder layer counts
- different feed-forward dimensions
- different dropout rates

## Evaluate a Saved Model

Evaluate on the test split by default:

```bash
python source/evaluate.py --model-type baseline
python source/evaluate.py --model-type distilbert
python source/evaluate.py --model-type scibert
python source/evaluate.py --model-type custom_transformer
```

Evaluate another split:

```bash
python source/evaluate.py --model-type custom_transformer --split validation
```

Metrics include:

- accuracy
- weighted F1-score
- macro F1-score
- classification report
- confusion matrix
- class-level error analysis

## Predict a Field

```bash
python source/predict.py --model-type baseline --title "Graph neural networks for molecules" --abstract "We introduce a message passing architecture for molecular property prediction."
```

Available model types:

- `baseline`
- `distilbert`
- `scibert`
- `custom_transformer`

The script prints the predicted field and class probabilities.

## Interpreting the Comparison

Use the baseline to establish a strong classical reference point. It often performs well when field-specific vocabulary is highly distinctive.

DistilBERT should improve contextual understanding while staying relatively fast. SciBERT may outperform DistilBERT because its pretraining data is closer to arXiv-style scientific text.

The custom Transformer demonstrates architectural understanding. It may not beat pretrained models without much more data, compute, and tuning, but it is valuable for explaining how attention, positional encoding, depth, heads, feed-forward size, and dropout affect performance.

Weighted F1 is useful when class counts are imbalanced. Macro F1 is stricter because it treats all classes equally. The confusion matrices and class-level error analysis help identify which academic fields are being confused and whether the model is overfitting to dominant classes.

## Project Structure

```text
scientific-abstract-transformer-study/
├── source/
│   ├── explore_dataset.py
│   ├── prepare_dataset.py
│   ├── train_baseline.py
│   ├── train_distilbert.py
│   ├── train_scibert.py
│   ├── train_custom_transformer.py
│   ├── transformer_model.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── run_ablation.py
│   ├── hf_training.py
│   └── utils.py
├── data/processed/
├── results/
├── figures/
├── saved_models/
├── requirements.txt
└── README.md
```
