# Expanded preregistered study protocol

## Claim boundary
This is a **planned empirical study**, not an executed competition submission. All numeric baseline values in this document are published organizer numbers, not this team's measurements. All effectiveness, timing, memory, and compute results for the proposed methods remain **TBD** until trained and evaluated.

## Research question and hypotheses
- **RQ1:** Does a modified single-encoder, two-output model improve mean F1 relative to a reproduced CodeBERT baseline? H1: tuned variant mean F1 > reproduced baseline mean F1 on public evaluation data.
- **RQ2:** Do separately selected decision thresholds improve the SATD F1 over a common 0.5 threshold? H2: internal-validation-chosen thresholds improve SATD F1 without unacceptable weakness F1 loss.
- **RQ3:** How do maximum input length and model family affect performance and inference cost? H3: shorter inputs reduce prediction latency with a possible loss in mean F1.
- **RQ4:** Which misclassifications are most common and are they associated with class prevalence, truncation, or project overlap? Exploratory; no directional hypothesis.

## Unit of analysis
One formatted source-code function with associated comments and one vector of binary labels `[weakness, MAT]`. The two targets are defined at the instance level; report each separately and their arithmetic mean. Do not confuse SATD (the competition task) with free-form comment generation (the team's original idea).

## Data and partitions
- Official Hugging Face dataset: `NLBSE/nlbse27-code-comment-classification`.
- Organizer repository lists 214,586 public training records and 30,655 public evaluation records; an additional 61,311 examples are hidden for final competition ranking. Treat these as published counts, and confirm your downloaded version before reporting.
- Public **train** only: create an internally stratified 85% fit / 15% validation split using random seed 42. If joint-label groups contain too few examples, document the fallback and its consequences.
- Public evaluation set: **once**, after model design, learning rate, loss weighting, thresholds, and max sequence length have been selected and frozen. Never search it for better thresholds. Organizer wording calls it a validation set, while the hub names it `test`.
- Run `python -m src.audit_dataset` to record joint-label distribution, exact-function duplicates, conflicts, and project overlap. Report potential leakage as a threat; an exact overlap count is not automatically proof that organizers made an error.

## Methods and planned variants
1. **Naive control (optional):** always predict the most frequent label on training. Not a competition candidate; establishes data imbalance.
2. **TF-IDF+logistic:** one shared TF-IDF representation and two task classifiers. Useful speed control, but two fitted heads should be described transparently; they share features rather than a jointly optimized neural encoder.
3. **CodeBERT reproduction:** `microsoft/codebert-base`, two-label sigmoid head, binary cross-entropy loss, same fit/validation split as variants. Use same software and GPU for head-to-head tests.
4. **Threshold-only:** freeze CodeBERT model weights; choose separate thresholds for weakness and SATD using *internal* validation predictions. Compare to a fixed 0.50 threshold on that internal set and carry one choice forward.
5. **Length ablation:** compare maximum sequence lengths 128, 256 and 512 with an otherwise fixed setup. Choose length with internal held-out data; input truncation may remove relevant comments.
6. **Class-weighted loss (needs implementation):** derive per-label positive weighting from fitting split only, apply to `BCEWithLogitsLoss`, and keep all other settings fixed. Do **not** claim this was run unless you implement and run it. Record why this option was dropped if hardware/time prevent it.
7. **Seed sensitivity (if feasible):** seeds 42, 52, 62 for finalist versus reproduced baseline, mean and standard deviation. Do not present one run as a confidence interval.

## Metrics
For each target separately: TP, FP, FN, TN, precision, recall, F1, and accuracy. Use `zero_division=0` and explain when this applies. Main effectiveness metric: `(F1_weakness + F1_SATD)/2`. Also log inference seconds per example for a fixed batch size, GPU/CPU model, peak device memory when available, and train time. Organizer's final score is not just F1: approximately 60% mean F1 + 20% execution time + 20% GFLOPs under its own evaluator; do not compute a purported official ranking without executing the organizer notebook.

## Model selection and leakage prevention
- Run all hyperparameter searches solely on internal validation records derived from **train**.
- Use deterministic split, fixed seeds, and a structured log for every run.
- Fit the vectorizer, tokenizer-dependent choices, class weights, and model solely using fit split; pretrained CodeBERT is an external initialization, not a data leakage event by itself.
- If duplicates in fit/internal-validation are substantial, document a group-aware alternative as robustness analysis. Project-held-out data assesses transfer but is not necessarily comparable to official splits.
- Freeze a final configuration in `results/final_config.json`, with commit hash and package freeze, before official public evaluation.
- Never choose one of many runs *after observing public evaluation performance* and call it preselected.

## Error analysis protocol
Select up to 20 false positives and 20 false negatives per label using a **random fixed seed**, and document: task, prediction confidence, code truncation yes/no, descriptive code pattern, and plausible error mechanism. Quote only short, license-safe snippets; remove secrets and identifiers. Separately report frequency of errors by project and joint label. A human interpretation is a hypothesis, not proven causal evidence. Make a table with at least 4 representative cases; do not fabricate examples and present them as observations.

## Statistical reporting
For a single fixed public evaluation set, use paired bootstrap resampling of instances (e.g., 1,000 replicates) for a descriptive 95% interval of the *difference in mean F1* between two frozen models. Models must be evaluated on the same rows. If public-set-based intervals are used, do not treat them as validation for model selection. A paired interval does not control project-level dependency; consider project-cluster bootstrap as an additional check if project IDs are available.

## Threats to validity
- **Construct:** weakness labels may not represent exploitable vulnerabilities; SATD labels may not capture all debt.
- **Internal:** differences in hyperparameter budgets, seeds, batch sizes, and hardware may confound outcomes.
- **External:** training data from selected open-source C/C++ projects may not generalize to every language or repository.
- **Conclusion:** minority class prevalence, duplicated functions, and a single public evaluation split can make score differences fragile.
- **Reproducibility:** dependency updates, unavailable checkpoints, and different GPU versions affect exact results. Store environment freeze, revision, seeds, and commands.

## Stop/go decisions
- If downloading the dataset fails, **do not invent scores**. Report environment blocker and run unit tests only.
- If a GPU is unavailable, prioritize TF-IDF and a small CodeBERT smoke run; label all limited-sample results as development tests.
- If no candidate beats the baseline, a defensible research outcome is a negative result plus a measured speed/accuracy tradeoff.
- If the class requires the originally approved code-comment **generation** task, change tracks and do not submit classifier results under that name without the instructor's approval.
