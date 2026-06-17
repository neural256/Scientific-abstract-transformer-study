# Scientific Abstract Field Classification

This project compares classical machine learning, pretrained Transformer models, and a custom Transformer encoder for classifying scientific paper abstracts into broad academic fields.

The task is a five-class text classification problem: given a paper title and abstract, predict whether it belongs to Computer Science, Mathematics, Physics, Biology, or Economics.

## Highlights

- Prepared the `TimSchopf/arxiv_categories` Hugging Face dataset into train, validation, and test CSV files.
- Implemented a strong TF-IDF + Logistic Regression baseline.
- Fine-tuned DistilBERT using Hugging Face Transformers and PyTorch.
- Added a SciBERT fine-tuning script for scientific-domain pretraining comparison.
- Built a custom Transformer encoder manually in PyTorch without using `torch.nn.TransformerEncoder`.
- Added reusable evaluation, plotting, prediction, model saving, and ablation workflows.

## Dataset

The dataset is loaded from Hugging Face:

```text
TimSchopf/arxiv_categories
```

Raw columns:

- `id`
- `title`
- `abstract`
- `categories`
- `creation_date`

The processed model input joins the title and abstract:

```text
title [SEP] abstract
```

Processed files are generated under `data/processed/`, but the CSV files are intentionally not committed because they can be recreated from the public dataset.

## Label Mapping

The original arXiv categories contain archive paths such as:

```text
Computer Science Archive->cs.CV
Physics Archive->astro-ph->astro-ph.EP
Mathematics Archive->math.PR
```

This project maps the archive prefix to five broad labels:

| Archive prefix | Label |
| --- | --- |
| `Computer Science Archive` | Computer Science |
| `Mathematics Archive` | Mathematics |
| `Physics Archive` | Physics |
| `Quantitative Biology Archive` | Biology |
| `Economics Archive` | Economics |

## Models

### TF-IDF + Logistic Regression

The baseline uses TF-IDF features with unigram and bigram terms, followed by a balanced one-vs-rest Logistic Regression classifier. It is fast, interpretable, and useful as a reference point for the neural models.

### DistilBERT

DistilBERT is fine-tuned with a sequence classification head. It provides a compact pretrained Transformer baseline with lower compute cost than BERT-base models.

### SciBERT

SciBERT is included because it is pretrained on scientific text. It is expected to be a strong comparison point for scientific abstract classification, especially when trained on enough examples.

### Custom Transformer Encoder

The custom model in `source/transformer_model.py` implements the main Transformer encoder components directly:

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

The implementation uses PyTorch tensor operations and standard layers such as `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, and `nn.Dropout`, but does not use `torch.nn.TransformerEncoder` as the main model.

## Current Results

The committed metrics are saved in `results/`. The table below reports test-set results from the current completed runs.

| Model | Accuracy | Weighted F1 | Macro F1 |
| --- | ---: | ---: | ---: |
| TF-IDF + Logistic Regression | 0.9476 | 0.9481 | 0.8135 |
| Custom Transformer Encoder | 0.9170 | 0.9134 | 0.5395 |
| DistilBERT | 0.9170 | 0.9130 | 0.5398 |

The baseline is currently the strongest run by weighted F1. This is a useful result rather than a failure: field classification often has strong vocabulary signals, so a TF-IDF baseline can be very competitive. The pretrained and custom Transformer runs still provide important architectural comparisons and can be improved with longer training, larger sample sizes, and hyperparameter tuning.

Confusion matrices are available in `figures/`:

- `figures/confusion_matrix_baseline.png`
- `figures/confusion_matrix_custom_transformer.png`
- `figures/confusion_matrix_distilbert.png`

The ablation results are saved as:

- `results/ablation_results.csv`
- `figures/ablation_plot.png`

## Ablation Study

The custom Transformer ablation compares:

- positional encoding vs no positional encoding
- different numbers of attention heads
- different numbers of encoder layers
- different feed-forward dimensions
- different dropout rates

These experiments are intended to show how individual Transformer design choices affect validation performance.

## Installation

Use Python 3.11.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux:

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

## Training

All commands are run from the project root.

Train the baseline:

```bash
python source/train_baseline.py
```

Train the custom Transformer:

```bash
python source/train_custom_transformer.py
```

Train DistilBERT:

```bash
python source/train_distilbert.py
```

Train SciBERT:

```bash
python source/train_scibert.py
```

For faster laptop runs, each neural training script supports sample limits:

```bash
python source/train_distilbert.py --max-train-samples 1000 --max-val-samples 300 --max-test-samples 300 --epochs 1 --batch-size 4 --max-length 128
```

For stronger runs, use the full processed splits:

```bash
python source/train_distilbert.py --max-train-samples 0 --max-val-samples 0 --max-test-samples 0 --epochs 3
```

## Evaluation

Evaluate a saved model on the test split:

```bash
python source/evaluate.py --model-type baseline
python source/evaluate.py --model-type distilbert
python source/evaluate.py --model-type scibert
python source/evaluate.py --model-type custom_transformer
```

Metrics include:

- accuracy
- weighted F1-score
- macro F1-score
- classification report
- confusion matrix
- class-level error analysis

## Prediction

Predict the field for a new title and abstract:

```bash
python source/predict.py --model-type baseline --title "Graph neural networks for molecules" --abstract "We introduce a message passing architecture for molecular property prediction."
```

Available model types:

- `baseline`
- `distilbert`
- `scibert`
- `custom_transformer`

## Project Structure

```text
scientific-abstract-transformer-study/
|-- source/
|   |-- explore_dataset.py
|   |-- prepare_dataset.py
|   |-- train_baseline.py
|   |-- train_distilbert.py
|   |-- train_scibert.py
|   |-- train_custom_transformer.py
|   |-- transformer_model.py
|   |-- evaluate.py
|   |-- predict.py
|   |-- run_ablation.py
|   |-- hf_training.py
|   `-- utils.py
|-- data/processed/
|-- results/
|-- figures/
|-- saved_models/
|-- requirements.txt
`-- README.md
```

## Repository Notes

The repository tracks source code, documentation, final metrics, and final plots. Large generated files are excluded:

- processed dataset CSV files
- saved model weights
- virtual environments
- smoke-test outputs

The `.gitkeep` files are placeholders that allow Git to preserve empty directories such as `data/processed/` and `saved_models/`.
