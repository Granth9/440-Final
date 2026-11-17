# Three-minute presentation outline (edit after experiments)

**0:00–0:20 — Grant, problem:** "We studied whether a multitask AI system can identify potential weaknesses in code and self-admitted technical debt from source-code functions and their comments. Our reference point was the published CodeBERT baseline."

**0:20–0:45 — Ryley, dataset:** "The organizer provided a publicly available training and evaluation set, with a held-back final test. Each example has a function and a two-bit label. We also audited label balance, duplicates, and potential data overlap." Show `src/audit_dataset.py` and actual audit JSON, not made-up numbers.

**0:45–1:25 — Grant, implementation:** Show `src/common.py`, `src/train_tfidf.py`, `src/train_transformer.py`. Explain the two-label output, `BCEWithLogitsLoss`/sigmoid, held-out thresholds, and reproducibility setup. If a planned variant was not implemented, don't say it was.

**1:25–2:10 — Ryley, measured results:** Show a table from `results/comparison.csv` with **actual** per-task precision, recall, F1, mean F1 and runtime. Explain whether the result did or did not beat the 0.910 published mean F1. Distinguish a controlled re-run from an organizer-published comparison.

**2:10–2:40 — Grant, interpretation:** Identify the most common real error patterns, fairness of comparison, limitations and tradeoffs. Show one real misclassified example with identifiers removed.

**2:40–3:00 — Both, conclusion:** "Our main finding was [INSERT ACTUAL FINDING]. Next, we would test [INSERT NEXT STEP]." Point to the GitHub repository and reproducibility commands.

## On-screen demo order
1. `README.md` and dependency installation.
2. `pytest -q` showing passing tests.
3. `python -m src.audit_dataset` and its saved JSON (if downloaded).
4. One model training command, showing saved model and results (recorded beforehand if it takes too long).
5. `results/comparison.csv` and one meaningful code or comment error.
6. Final summary and responsibilities. Both presenters' faces on screen when presenting, consistent with course instructions.
