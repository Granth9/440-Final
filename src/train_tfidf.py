"""Cheap comparison model: shared text representation, two logistic heads."""
import argparse
from pathlib import Path
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from src.common import load_data, unpack, partition_train, choose_thresholds, metrics, write_json

def fit_model(examples, max_features=80000):
    x, y = unpack(examples)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2,
                                  max_features=max_features, sublinear_tf=True, dtype=np.float32)),
        ("model", MultiOutputClassifier(LogisticRegression(max_iter=600, class_weight="balanced")))
    ])
    pipeline.fit(x, y)
    return pipeline

def probabilities(model, code):
    return np.column_stack([part[:, 1] for part in model.predict_proba(code)])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-limit", type=int)
    p.add_argument("--test-limit", type=int)
    p.add_argument("--output", default="models/tfidf.joblib")
    p.add_argument("--evaluate-public-test", action="store_true", help="Do this once after tuning is frozen")
    args = p.parse_args()
    train, test = load_data(args.train_limit, args.test_limit)
    fit, tune = partition_train(train)
    model = fit_model(fit)
    xt, yt = unpack(tune)
    cutoffs = choose_thresholds(yt, probabilities(model, xt))
    tuning_metrics = metrics(yt, (probabilities(model, xt) >= cutoffs).astype(int))
    print("Internal tuning:", tuning_metrics)
    # IMPORTANT: retrain on all public TRAIN after selecting thresholds.
    final_model = fit_model(train)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "thresholds": cutoffs}, out)
    write_json("results/tfidf_tuning.json", {"thresholds": cutoffs, "metrics": tuning_metrics,
       "limited_run": args.train_limit is not None})
    if args.evaluate_public_test:
        xpub, ypub = unpack(test)
        yp = (probabilities(final_model, xpub) >= cutoffs).astype(int)
        result = metrics(ypub, yp)
        write_json("results/tfidf_public_test.json", result)
        print("Public test:", result)
    else:
        print("Public test not evaluated; run again with --evaluate-public-test after model decisions are frozen")

if __name__ == "__main__":
    main()
