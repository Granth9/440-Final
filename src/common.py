"""Shared schema checks, metrics, and reproducible data utilities."""
import json
import random
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

DATASET_ID = "NLBSE/nlbse27-code-comment-classification"
TASKS = ("weakness", "MAT")
OFFICIAL_BASELINE_F1 = {"weakness": 0.939, "MAT": 0.881}

def check_examples(examples):
    if not examples:
        raise ValueError("No examples found")
    if not all("Function" in x and "Label" in x for x in examples):
        raise ValueError("Dataset must contain Function and Label fields")
    for x in examples:
        if not isinstance(x["Function"], str) or len(x["Label"]) != 2:
            raise ValueError("Invalid Function or Label")
        if any(int(v) not in (0, 1) for v in x["Label"]):
            raise ValueError("Labels must be binary")

def unpack(examples):
    check_examples(examples)
    return [x["Function"] for x in examples], np.asarray([x["Label"] for x in examples], dtype=int)

def metrics(y_true, y_pred):
    true = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    if true.ndim != 2 or true.shape[1] != 2 or true.shape != pred.shape:
        raise ValueError("Expected matching (n,2) arrays")
    result = {}
    for i, name in enumerate(TASKS):
        p, r, f1, _ = precision_recall_fscore_support(true[:, i], pred[:, i], average="binary", zero_division=0)
        result[name] = {"precision": float(p), "recall": float(r), "f1": float(f1),
                        "accuracy": float(accuracy_score(true[:, i], pred[:, i])),
                        "positives": int(true[:, i].sum())}
    result["mean_f1"] = float(sum(result[t]["f1"] for t in TASKS)/2)
    result["baseline_mean_f1"] = float(sum(OFFICIAL_BASELINE_F1.values())/2)
    result["delta_vs_published_baseline"] = result["mean_f1"] - result["baseline_mean_f1"]
    return result

def choose_thresholds(y_true, probabilities):
    """Select independent thresholds on held-out tuning data only."""
    from sklearn.metrics import f1_score
    y = np.asarray(y_true)
    probs = np.asarray(probabilities)
    if y.shape != probs.shape or y.ndim != 2 or y.shape[1] != 2:
        raise ValueError("Expected matching (n,2) matrices")
    candidates = np.arange(0.10, 0.901, 0.05)
    return [float(max(candidates, key=lambda x: (f1_score(y[:, i], probs[:, i] >= x, zero_division=0), -abs(x-0.5)))) for i in range(2)]

def partition_train(examples, validation_fraction=0.15, seed=42):
    """Splits PUBLIC TRAIN only; does not touch official public test labels."""
    labels = [str(tuple(x["Label"])) for x in examples]
    counts = {k: labels.count(k) for k in set(labels)}
    stratify = labels if all(v >= 2 for v in counts.values()) else None
    return train_test_split(list(examples), test_size=validation_fraction, random_state=seed, stratify=stratify)

def write_json(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

def load_data(train_limit=None, test_limit=None):
    from datasets import load_dataset
    dataset = load_dataset(DATASET_ID)
    if "train" not in dataset or "test" not in dataset:
        raise ValueError("Official hub layout changed; inspect dataset split names")
    tr = dataset["train"]
    te = dataset["test"]
    if train_limit is not None:
        tr = tr.shuffle(seed=42).select(range(min(train_limit, len(tr))))
    if test_limit is not None:
        te = te.shuffle(seed=42).select(range(min(test_limit, len(te))))
    return [dict(x) for x in tr], [dict(x) for x in te]
