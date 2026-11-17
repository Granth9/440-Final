# NLBSE 2027 — Beating the Multitask Code Classification Baseline

**Team:** Grant Hernandez and Ryley Wickstrom  
**Course:** CS480A7 Empirical Software Engineering  
**Status:** Research plan, implementation starter and report draft. **No trained competition results are claimed.**

## Important topic correction
Our initial approved idea was to *generate* better code comments. That is NOT the 2027 competition task: NLBSE 2027 requires **classifying functions into two labels** (vulnerability/weakness, self-admitted technical debt). We recommend submitting a revised topic for approval; see `docs/topic_submission.md`. If the instructor insists on comment generation, use `docs/alternate_original_topic.md` and treat it as a separate proposed research paper, not an NLBSE 2027 competition entry.

## Research question
Can a modified multitask model improve over the NLBSE 2027 CodeBERT baseline on average F1, without unacceptable runtime/computational cost?

## Official material
- Rules: https://nlbse2027.github.io/tools/
- Code and baseline: https://github.com/nlbse2027/code-comment-classification
- Dataset: https://huggingface.co/datasets/NLBSE/nlbse27-code-comment-classification
- Competition notebook: https://colab.research.google.com/drive/1b_EiXx5woyCrDntwbljfi-BwS9Dwvhoa?usp=sharing
- Organizer baseline (published): F1 weakness 0.939; F1 MAT 0.881; mean 0.910.

## Project tree
```
NLBSE2027_Project/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── common.py                # load data, schema, 2-task metrics, threshold choice
│   ├── train_tfidf.py          # CPU-friendly shared-vectorizer 2-head comparison
│   ├── train_transformer.py    # CodeBERT 2-output AI model
│   └── predict.py              # prediction on source code (TF-IDF)
├── tests/test_common.py
├── docs/
│   ├── research_and_methods.md
│   ├── experiment_plan.md
│   ├── timeline_and_tasks.md
│   ├── topic_submission.md
│   ├── alternate_original_topic.md
│   └── submission_checklist.md
├── report/report_draft.md      # course-required sections; results are placeholders
├── results/                    # generated when experiments run
└── models/                     # generated model files
```

## Setup (Python 3.10–3.12 recommended)
```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
python -m pytest -q
```
**Internet is needed** the first time to download the official data and CodeBERT weights. A GPU is recommended for CodeBERT training; CPU training can be slow.

## Run the lightweight experiment first
A brief smoke run on **random samples** (not a valid final benchmark):
```bash
python -m src.train_tfidf --train-limit 2000 --test-limit 200
```
Full public training run, without reading the official public test set for model selection:
```bash
python -m src.train_tfidf
```
After the feature choices and thresholds are frozen, run the **public test** exactly once:
```bash
python -m src.train_tfidf --evaluate-public-test
```
Results are written to `results/tfidf_tuning.json` and `results/tfidf_public_test.json`.

## Run the AI model
Small **pipeline-only** run:
```bash
python -m src.train_transformer --train-limit 500 --test-limit 100 --epochs 1 --batch-size 4
```
Full training:
```bash
python -m src.train_transformer --epochs 2 --batch-size 8 --max-length 256
```
Choose hyperparameters with INTERNAL held-out training data only. Finally, run one official public-test evaluation with the frozen choices:
```bash
python -m src.train_transformer --epochs 2 --batch-size 8 --max-length 256 --evaluate-public-test
```
**Important:** For a scientifically strict one-time public test, train and freeze your final model first and add a separate load-and-evaluate stage, or retain the same config and seed; rerunning with these commands retrains a model. Record its resulting versions and checkpoint. The supplied transformer implementation is a starter experiment, NOT the organizer's exact scoring interface or a trained submission artifact. For a real competition entry, adapt/verify the official Colab inference adapter and GFLOPs procedure.

## Predict on a code snippet with the trained lightweight model
```bash
python -m src.predict --file example_function.txt
# or
python -m src.predict --code 'int foo(){return 1;}'
```
Labels returned in `[weakness, MAT]` order. Only load `.joblib` model files you trust.

## Evaluation protocol
1. Use public `train` (214,586 items, according to the organizer repository) for training and INTERNAL validation/threshold selection.
2. Use official public `test` (30,655 items, described as a public evaluation set in repository) only after choices are frozen. The organizers hold back a separate 61,311-instance hidden test set.
3. Report F1, precision, recall, accuracy for each task; mean F1; timing; peak memory/hardware; thresholds; random seed.
4. Compare against published official F1s, but **do not claim a controlled comparison** until same hardware, data, procedure, and organizer scoring are checked.
5. Compare base CodeBERT and tuned CodeBERT under identical settings, plus TF-IDF control. Ablate thresholds, sequence length and class-loss handling (if implemented).
6. Check for leakage, duplicated source functions, project imbalance, label imbalance, and error examples. Never train or tune on the organizers' public test.

## Reproducibility and submission
- Insert your **actual GitHub repo URL** here once created: `https://github.com/REPLACE_WITH_YOUR_ACCOUNT/REPLACE_WITH_YOUR_REPO`
- Record exact package versions via `python -m pip freeze > environment_freeze.txt` after running.
- Complete the paper and measured results; the report draft has clearly marked placeholders.
- Zip **this directory**, plus checkpoint or a stable download reference if the model is too large. Include `requirements.txt` and trained artifacts needed for inference.
- Use the venue's final 2–4 page style for external entry, but the **course requires at least 4 pages**; references/page constraints can conflict, so maintain separate course and competition versions.
- Send group topic through course spreadsheet and Discord. We cannot submit to those external accounts from this file.

## References
Official competition rules, competition repository, Mock et al. (2024) MADE-WIC, Feng et al. (2020) CodeBERT, Russo et al. (2025) VulSATD. Full references and links are in `docs/research_and_methods.md`.

## Expanded research package (v2)
The second package includes a protocol with hypotheses and validity threats, a dataset audit, cross-model CSV builder, results tables, a short presentation script and verified sources:
- `docs/expanded_study_protocol.md` — detailed experiment design, validation, leakage, statistical inference and ablations
- `docs/results_template.md` — final result tables and evidence-aware language
- `docs/presentation_script.md` — both-person 3-minute presentation and code demo
- `docs/references_verified.md` — official requirements and academic sources
- `src/audit_dataset.py` — label frequencies, duplicate function hashes, cross-split overlap
- `src/compare_results.py` — convert measured JSON experiment files to one CSV

Run additional steps:
```bash
python -m src.audit_dataset --train-limit 3000 --test-limit 500
# For final analysis: python -m src.audit_dataset
python -m src.compare_results results/tfidf_public_test.json results/codebert_public_test.json
```
These tools **do not generate model predictions or substitute for training**. The comparison command requires existing result files produced by real experiments. When deciding whether to include any optional ablation, implement and test it first; the protocol identifies unimplemented variants explicitly.

## Full execution blueprint
See `PROJECT_MASTER_OUTLINE.md` for the complete detailed 17-step plan, research questions, experimental controls, paper section outline, table shells, team assignments and submission checklist.
