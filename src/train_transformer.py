"""Fine-tune one shared CodeBERT encoder with two binary outputs (multilabel).

Training protocol:
1. Fit on 85% of the public training split.
2. Select one decision threshold per task on the held-out 15%.
3. Reinitialize CodeBERT and refit on 100% of the public training split.
4. Save the full-data model and thresholds selected in step 2.
5. Optionally evaluate once on the public evaluation split.
"""
import argparse
import time

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.common import (
    choose_thresholds,
    load_data,
    metrics,
    partition_train,
    unpack,
    write_json,
)

MODEL_ID = "microsoft/codebert-base"


def prep(raw, tokenizer, max_length):
    """Tokenize examples while retaining the two binary target labels."""
    def batch(examples):
        encoded = tokenizer(
            examples["Function"],
            truncation=True,
            max_length=max_length,
        )
        encoded["labels"] = [
            [float(value) for value in label] for label in examples["Label"]
        ]
        return encoded

    dataset = Dataset.from_list(
        [
            {"Function": example["Function"], "Label": list(example["Label"])}
            for example in raw
        ]
    )
    return dataset.map(batch, batched=True, remove_columns=dataset.column_names)


def predict_probabilities(trainer, dataset):
    """Return sigmoid probabilities for both independent classification heads."""
    output = trainer.predict(dataset)
    logits = np.asarray(output.predictions)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))


def make_model(model_name):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        problem_type="multi_label_classification",
    )


def make_trainer(model, args, tokenizer, collator, run_dir, train_dataset):
    training_args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_steps=250,
        save_strategy="no",
        report_to="none",
        seed=42,
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=torch.cuda.is_available(),
    )
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--output-dir", default="models/codebert_multitask")
    parser.add_argument(
        "--evaluate-public-test",
        action="store_true",
        help="Evaluate once on the public evaluation split after training choices are frozen.",
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    train_examples, public_examples = load_data(args.train_limit, args.test_limit)
    fit_examples, tune_examples = partition_train(train_examples)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Stage 1: fit only on the fitting partition to calibrate thresholds.
    fit_dataset = prep(fit_examples, tokenizer, args.max_length)
    tune_dataset = prep(tune_examples, tokenizer, args.max_length)
    tuning_model = make_model(args.model)
    tuning_trainer = make_trainer(
        tuning_model,
        args,
        tokenizer,
        collator,
        args.output_dir + "_threshold_tuning",
        fit_dataset,
    )

    tuning_start = time.perf_counter()
    tuning_trainer.train()
    threshold_fit_seconds = time.perf_counter() - tuning_start

    y_tune = unpack(tune_examples)[1]
    tuning_probabilities = predict_probabilities(tuning_trainer, tune_dataset)
    thresholds = choose_thresholds(y_tune, tuning_probabilities)
    tuning_metrics = metrics(
        y_tune,
        (tuning_probabilities >= thresholds).astype(int),
    )

    # Stage 2: initialize a fresh model and refit on all public training data.
    # Thresholds remain those selected on the separate tuning partition.
    full_train_dataset = prep(train_examples, tokenizer, args.max_length)
    del tuning_trainer, tuning_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    final_model = make_model(args.model)
    final_trainer = make_trainer(
        final_model,
        args,
        tokenizer,
        collator,
        args.output_dir + "_full_training",
        full_train_dataset,
    )

    full_fit_start = time.perf_counter()
    final_trainer.train()
    full_fit_seconds = time.perf_counter() - full_fit_start

    final_trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    write_json(args.output_dir + "/thresholds.json", {"thresholds": thresholds})

    tuning_record = {
        "thresholds": thresholds,
        "metrics": tuning_metrics,
        "config": vars(args),
        "limited_run": args.train_limit is not None,
        "threshold_fit_examples": len(fit_examples),
        "threshold_tuning_examples": len(tune_examples),
        "final_training_examples": len(train_examples),
        "threshold_fit_training_seconds": threshold_fit_seconds,
        "full_data_training_seconds": full_fit_seconds,
        "final_model_refit_on_full_training_split": True,
    }
    write_json("results/codebert_tuning.json", tuning_record)

    print("Internal tuning:", tuning_metrics)
    print("Selected thresholds:", thresholds)
    print("Threshold-model training seconds:", round(threshold_fit_seconds, 2))
    print("Full-data model training seconds:", round(full_fit_seconds, 2))
    print("Saved final full-data model:", args.output_dir)

    if args.evaluate_public_test:
        public_dataset = prep(public_examples, tokenizer, args.max_length)
        public_start = time.perf_counter()
        public_probabilities = predict_probabilities(final_trainer, public_dataset)
        inference_seconds = time.perf_counter() - public_start
        public_metrics = metrics(
            unpack(public_examples)[1],
            (public_probabilities >= thresholds).astype(int),
        )
        public_metrics["inference_seconds_per_example"] = (
            inference_seconds / len(public_examples)
        )
        public_metrics["note"] = (
            "Public evaluation performed with the final model refit on all public "
            "training examples; thresholds were selected only on the internal "
            "training split. Do not use these results to tune the model."
        )
        write_json("results/codebert_public_test.json", public_metrics)
        print("Public test:", public_metrics)
    else:
        print(
            "Public test not evaluated. Freeze model choices before running "
            "again with --evaluate-public-test."
        )


if __name__ == "__main__":
    main()
