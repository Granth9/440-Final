# NLBSE 2027 Multitask Code Classification

**Team:** Grant Hernandez and Ryley Wickstrom  
**Course:** CS480A7 — Empirical Software Engineering  
**Repository:** https://github.com/Granth9/440-Final

## Project overview

This project investigates multitask classification of source-code functions for two labels in the NLBSE 2027 Code Comment Classification benchmark:

- **Weakness:** whether a function is labeled as containing a software weakness.
- **MAT:** whether a function is labeled as containing self-admitted technical debt.

The research question is whether a lightweight character-level TF-IDF model or a fine-tuned shared-encoder CodeBERT model provides a stronger quality/cost tradeoff relative to the published organizer baseline.

The output label order throughout this project is **`[weakness, MAT]`**. The two labels are independent; both may be positive for the same function.

## Dataset and evaluation protocol

Dataset: [NLBSE/nlbse27-code-comment-classification](https://huggingface.co/datasets/NLBSE/nlbse27-code-comment-classification).

The downloaded dataset contains 214,586 public training examples and 30,655 public evaluation examples. The organizers also describe a separate hidden test set. The project uses an internal 85/15 split of the public training data to select one decision threshold per label. Each final model is then refit on the full public training split. The public evaluation split is used only after model choices and thresholds are frozen.

**The public evaluation has already been run for this project. Do not use its labels or results to select new hyperparameters, thresholds, or model variants.** The saved public-evaluation results are in `results/`.

## Models

### TF-IDF comparison model

- Character-level TF-IDF features using 3–5 character n-grams.
- `min_df=2`, sublinear term frequency, up to 80,000 features.
- `MultiOutputClassifier` with class-balanced logistic regression (`max_iter=600`).
- Separate decision thresholds selected on the internal validation split.

This is a lightweight shared-feature, two-output comparison model. It is not a jointly optimized neural multitask encoder.

### CodeBERT model

- Base encoder: [`microsoft/codebert-base`](https://huggingface.co/microsoft/codebert-base).
- Two independent sigmoid outputs trained with multilabel binary cross-entropy.
- Two epochs, learning rate `2e-5`, batch size 8, maximum sequence length 256.
- Thresholds selected on the internal validation split, followed by a fresh-model refit on all 214,586 public training examples.

## Final public-evaluation results

F1 is the primary predictive metric. The organizer baseline values below are the published reference values, not a baseline model retrained by this team.

| Model | Weakness F1 | MAT F1 | Mean F1 |
|---|---:|---:|---:|
| Published organizer baseline | 0.9390 | 0.8810 | 0.9100 |
| TF-IDF + logistic regression | 0.8033 | **0.8371** | **0.8202** |
| Multitask CodeBERT | **0.8843** | 0.7114 | 0.7979 |

Additional public-evaluation metrics:

| Model / task | Precision | Recall | Accuracy |
|---|---:|---:|---:|
| TF-IDF — weakness | 0.7629 | 0.8481 | 0.8638 |
| TF-IDF — MAT | 0.8731 | 0.8040 | 0.9948 |
| CodeBERT — weakness | 0.8762 | 0.8926 | 0.9234 |
| CodeBERT — MAT | 0.9565 | 0.5663 | 0.9924 |

The TF-IDF model has the higher overall mean F1 in this experiment, while CodeBERT has the higher weakness F1. TF-IDF has the higher MAT F1, primarily due to substantially higher MAT recall. Neither model exceeds the published mean-F1 baseline.

Internal validation results are recorded separately in `results/tfidf_tuning.json` and `results/codebert_tuning.json`; they should not be confused with the public-evaluation results.

## Inference-time measurements

| Model | Public examples | Measured total | Seconds/example | Device |
|---|---:|---:|---:|---|
| TF-IDF | 30,655 | 20.805 s | 0.000679 s | CPU |
| CodeBERT | 30,655 | 134.169 s | 0.004377 s | CUDA GPU |

These are local measurements, not official competition runtime scores. The CodeBERT measurement includes tokenization, batching, GPU inference, and probability conversion; the TF-IDF measurement includes vectorization and prediction. The models were measured on different device types (CPU for TF-IDF and CUDA GPU for CodeBERT), so the observed timing ratio is not a hardware-independent comparison. The official competition cost calculation may also include metrics such as GFLOPs that this project did not measure.

## Dataset audit

The full audit is saved in `results/dataset_audit_full.json`. It reports:

- 808 duplicate rows and 31 conflicting-label function hashes in the training split.
- 24 duplicate rows and 1 conflicting-label function hash in the public evaluation split.
- 238 exact function hashes shared across the two splits.
- 15 project names represented in both splits.

These are descriptive audit findings. Shared hashes or project names alone do not prove leakage or establish that the split is invalid. The project distribution is highly imbalanced toward Linux Kernel examples, which limits claims about generalization to unseen projects.

## Repository structure

```text
440-Final/
├── README.md
├── requirements.txt
├── environment_freeze.txt       # exact local package snapshot, if included
├── .gitignore
├── src/
│   ├── common.py
│   ├── audit_dataset.py
│   ├── compare_results.py
│   ├── predict.py
│   ├── train_tfidf.py
│   └── train_transformer.py
├── tests/
├── models/
│   ├── tfidf_full.joblib
│   └── codebert_multitask/
└── results/
    ├── dataset_audit_full.json
    ├── tfidf_tuning.json
    ├── tfidf_public_test.json
    ├── codebert_tuning.json
    ├── codebert_public_test.json
    └── comparison.csv
```

The final trained artifacts are `models/tfidf_full.joblib` and the contents of `models/codebert_multitask/`. Intermediate smoke-test and threshold-tuning checkpoint folders are not required for inference.

## Setup

Tested with Python 3.11.9 on Windows and an NVIDIA GeForce RTX 4060 Laptop GPU for CodeBERT training/inference. A compatible CUDA-enabled PyTorch installation is required to reproduce GPU timing; CPU execution may be much slower.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -v
```

The first run may download the Hugging Face dataset and CodeBERT base weights. The saved final model directories are needed for local inference without retraining.

`environment_freeze.txt` records the exact package versions from the team's local environment. `requirements.txt` provides the project's version ranges.

## Use the saved TF-IDF model

Predict labels for a code snippet:

```powershell
python -m src.predict --code "int add(int a, int b) { return a + b; }"
```

Or read a function from a file:

```powershell
python -m src.predict --file example_function.txt
```

To specify another trusted model artifact:

```powershell
python -m src.predict --model models/tfidf_full.joblib --file example_function.txt
```

The JSON output contains `labels_weakness_MAT` and the two predicted probabilities. Only load trusted local Joblib files.

## Reproducing the experiments

The following commands retrain models and overwrite the corresponding model/tuning artifacts. They are provided for reproducibility; they are not necessary to use the saved final models.

Train the full TF-IDF model and select thresholds using the internal training validation split:

```powershell
python -m src.train_tfidf --output models/tfidf_full.joblib
```

Train CodeBERT, select thresholds internally, and refit on all public training examples:

```powershell
python -m src.train_transformer --epochs 2 --batch-size 8 --max-length 256 --output-dir models/codebert_multitask
```

The public-evaluation results in this repository were generated after these model choices were frozen. The public-evaluation commands are intentionally not presented as routine steps because this evaluation has already been consumed for the project. Do not use it for further tuning.

Run the full dataset audit:

```powershell
python -m src.audit_dataset
```

Rebuild the comparison CSV from the saved public-evaluation JSON files:

```powershell
python -m src.compare_results results/tfidf_public_test.json results/codebert_public_test.json
```

## Limitations

- Neither in-house model exceeded the published mean-F1 baseline.
- The two model families have different task-level strengths; the aggregate mean hides the CodeBERT MAT-recall weakness.
- The public evaluation split contains exact function-hash overlap with the training split and substantial project overlap; the audit does not establish whether this affects the scores.
- The data is dominated by a small number of projects, particularly Linux Kernel.
- Runtime measurements were taken on different device types and are not official competition cost scores.
- The project did not reproduce the organizer's complete scoring pipeline or report GFLOPs, so it does not claim an official competition ranking or a controlled reproduction of the organizer baseline.
- Results are from one seeded run; repeated-seed uncertainty estimates and cross-project generalization experiments were not completed.

## References

- NLBSE 2027 Tool Competition: https://nlbse2027.github.io/tools/
- Official benchmark repository: https://github.com/nlbse2027/code-comment-classification
- Dataset: https://huggingface.co/datasets/NLBSE/nlbse27-code-comment-classification
- CodeBERT: Feng et al. (2020), *Findings of EMNLP*. https://doi.org/10.18653/v1/2020.findings-emnlp.139
- MADE-WIC: Mock et al. (2024), *ASE*. https://doi.org/10.1145/3691620.3695348
- Multitask SATD and vulnerability detection: Russo et al. (2025), *ICPC*. https://doi.org/10.1109/ICPC66645.2025.00017
