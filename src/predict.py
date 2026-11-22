"""Generate two-label predictions with the saved TF-IDF model."""

import argparse
import json
from pathlib import Path

import joblib

from src.train_tfidf import probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="models/tfidf_full.joblib",
        help="Path to a trusted trained TF-IDF Joblib artifact",
    )
    parser.add_argument("--file", help="Read source code from a file")
    parser.add_argument("--code", help="Source code supplied directly")
    args = parser.parse_args()

    if bool(args.file) == bool(args.code is not None):
        parser.error("Provide exactly one of --file or --code")

    if args.file:
        source_path = Path(args.file)
        if not source_path.is_file():
            parser.error(f"Input file does not exist: {source_path}")
        code = source_path.read_text(encoding="utf-8")
    else:
        code = args.code

    model_path = Path(args.model)
    if not model_path.is_file():
        parser.error(
            f"Model file not found: {model_path}. "
            "Train the TF-IDF model or provide the correct --model path."
        )

    # Joblib can execute code while loading; only load trusted local artifacts.
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or not {"model", "thresholds"}.issubset(bundle):
        raise ValueError("Model artifact must contain 'model' and 'thresholds'.")

    predicted_probabilities = probabilities(bundle["model"], [code])[0]
    thresholds = bundle["thresholds"]
    if len(predicted_probabilities) != 2 or len(thresholds) != 2:
        raise ValueError("Expected exactly two outputs: weakness and MAT.")

    labels = [
        int(probability >= threshold)
        for probability, threshold in zip(predicted_probabilities, thresholds)
    ]
    print(
        json.dumps(
            {
                "labels_weakness_MAT": labels,
                "probabilities": predicted_probabilities.tolist(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
