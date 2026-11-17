"""Fine-tune one shared CodeBERT encoder with two binary outputs (multilabel)."""
import argparse
import time
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from src.common import load_data, partition_train, unpack, choose_thresholds, metrics, write_json

MODEL_ID = "microsoft/codebert-base"

def prep(raw, tokenizer, max_length):
    def batch(ex):
        encoded = tokenizer(ex["Function"], truncation=True, max_length=max_length)
        encoded["labels"] = [[float(v) for v in lab] for lab in ex["Label"]]
        return encoded
    ds = Dataset.from_list([{"Function": x["Function"], "Label": list(x["Label"])} for x in raw])
    return ds.map(batch, batched=True, remove_columns=ds.column_names)

def predict_probabilities(trainer, dataset):
    output = trainer.predict(dataset)
    logits = output.predictions
    return 1/(1+np.exp(-np.clip(logits, -40, 40)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--test-limit", type=int, default=None)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--learning-rate", type=float, default=2e-5)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--output-dir", default="models/codebert_multitask")
    ap.add_argument("--evaluate-public-test", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    tr, te = load_data(args.train_limit, args.test_limit)
    fit, tune = partition_train(tr)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    fitted = prep(fit, tokenizer, args.max_length)
    heldout = prep(tune, tokenizer, args.max_length)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, problem_type="multi_label_classification")
    training_args = TrainingArguments(
        output_dir=args.output_dir, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate, weight_decay=0.01,
        logging_steps=250, save_strategy="no", report_to="none", seed=42,
        fp16=torch.cuda.is_available())
    trainer = Trainer(model=model, args=training_args, train_dataset=fitted,
                      processing_class=tokenizer)
    trainer.train()
    yt = unpack(tune)[1]
    probs = predict_probabilities(trainer, heldout)
    thresholds = choose_thresholds(yt, probs)
    internal = metrics(yt, (probs >= thresholds).astype(int))
    write_json("results/codebert_tuning.json", {"thresholds": thresholds,
         "metrics": internal, "config": vars(args), "limited_run": args.train_limit is not None})
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    write_json(args.output_dir+"/thresholds.json", {"thresholds": thresholds})
    print("Internal tuning:", internal)
    if args.evaluate_public_test:
        public_set = prep(te, tokenizer, args.max_length)
        t0 = time.perf_counter()
        pp = predict_probabilities(trainer, public_set)
        elapsed = time.perf_counter() - t0
        pub = metrics(unpack(te)[1], (pp >= thresholds).astype(int))
        pub["inference_seconds_per_example"] = elapsed / len(te)
        pub["note"] = "Public test evaluated using a model fit only on internal training; tune only on internal held-out set"
        write_json("results/codebert_public_test.json", pub)
        print("Public test:", pub)
    else:
        print("Public test not evaluated. Freeze choices before --evaluate-public-test")

if __name__ == "__main__":
    main()
