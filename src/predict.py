"""Generate a two-value prediction from an already trained TF-IDF model."""
import argparse
import json
import joblib
from src.train_tfidf import probabilities

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/tfidf.joblib")
    p.add_argument("--file", help="Read source function from file instead of --code")
    p.add_argument("--code", help="Source function text")
    args = p.parse_args()
    if not args.file and args.code is None:
        p.error("Pass --file or --code")
    code = open(args.file, encoding="utf8").read() if args.file else args.code
    bundle = joblib.load(args.model)  # Load only trusted local model files
    probs = probabilities(bundle["model"], [code])[0]
    labels = [int(p >= cut) for p, cut in zip(probs, bundle["thresholds"])]
    print(json.dumps({"labels_weakness_MAT": labels, "probabilities": probs.tolist()}, indent=2))

if __name__ == "__main__":
    main()
