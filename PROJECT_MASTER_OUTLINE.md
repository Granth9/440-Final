# Complete project blueprint — NLBSE 2027 Code Classification

**Team:** Grant Hernandez; Ryley Wickstrom  
**Course:** CS480A7 Empirical Software Engineering  
**Working paper title:** *Can Simple Tuning Improve Multitask Code Classification? An Empirical Study of CodeBERT on Weakness and Technical-Debt Labels*  
**Status:** implementation starter and research protocol complete; full-dataset training, evaluated findings, repository publication, and instructor approval still required.  

## 0. Topic and scope decision

The team's previously proposed topic was *AI-generated code comment quality*, but the official NLBSE 2027 **Code Classification** competition does **not** generate comments. It predicts two binary labels for an existing source-code function: weakness/vulnerability and MAT (the dataset's maintenance/technical-debt annotation, discussed as SATD in competition materials). Thus this document is a **revised topic** requiring group spreadsheet and Discord updates and acceptance by the instructor. If approval is restricted to comment generation, see `docs/alternate_original_topic.md` and do not represent that alternative as a submission to this tool competition.

## 1. Main problem

Code review at scale is costly. A classifier could prioritize potentially vulnerable code and technical debt but may be too expensive to deploy. The NLBSE 2027 organizers supply a labeled MADE-WIC-derived dataset, a CodeBERT multitask baseline, and a common evaluation setting. Our contribution is to **reproduce a controlled reference and test two simple, transparent changes**: decision-threshold tuning and the amount of source code included. We additionally test an inexpensive TF-IDF baseline. We will avoid claiming advances in code-comment generation; this project is a classification study.

## 2. Research questions, hypotheses and observable variables

| RQ | Research question | Hypothesis | Measured evidence |
|---|---|---|---|
| RQ1 | Does a shared CodeBERT model outperform a TF-IDF model? | CodeBERT has higher mean task F1 but higher runtime | per-task precision/recall/F1, mean F1, inference s/example |
| RQ2 | Do label-specific thresholds outperform default 0.50 thresholds? | Yes, particularly MAT recall | per-task F1, confusion matrix, threshold values on internal development set |
| RQ3 | Does reducing sequence length improve efficiency enough to justify possible accuracy loss? | 128 tokens is faster, potentially less accurate than 256 or 512 | F1, tokens truncated, runtime and peak memory |
| RQ4 | What mistakes limit performance? | Errors cluster in ambiguous examples and poorly represented positives | manually reviewed false positive/negative samples, with project and length context |

**Independent variables:** model family (TF-IDF vs CodeBERT); classification thresholds (0.50 vs tuned); maximum token length (128, 256, 512); random seeds, if resources allow.  
**Dependent variables:** weakness precision/recall/F1; MAT precision/recall/F1; mean of two F1 values; inference seconds/example; peak RAM/VRAM; any organizer-provided GFLOPs estimate.  
**Control variables:** same dataset version and split, tokenization configuration within each comparison, random seeds, hardware, inference batch size, metrics implementation, and model checkpoints where only a threshold changes.

## 3. Official dataset — exact meaning of records

Official repository: https://github.com/nlbse2027/code-comment-classification  
Official Hugging Face ID: `NLBSE/nlbse27-code-comment-classification`  
Official reported participant splits: `train` 214,586 examples, `test` 30,655 examples (a public evaluation set). The organizers retain an additional hidden 61,311 examples. The competition web page calls the public evaluation data a *validation set*; the Hub/repository call it `test`. These are the same **participant-facing evaluation role**, not a second available private test set.

Fields: `Projectname`, `Filepath`, `Function`, `Label`. Pass **only `Function`** to the classifier; use project/path for descriptive statistics and error analysis, not predictors. `Label` is ordered `[weakness, MAT]`, where each is 0 or 1.

**Before training:** record dataset revision/date, sizes, task positive prevalence, joint-label counts, code length distribution, duplicate function hashes within and across splits, and overlap in `Projectname`. The prepared dataset's split is not necessarily project-disjoint, so discuss this limitation rather than silently assuming cross-project generalization.

## 4. Evaluation without leakage

1. Load official public `train` and public `test`/evaluation split.
2. Reserve ~15% of public `train` as an **internal tuning split** with fixed seed. Split by joint label when possible. This is not organizer's held-out public evaluation split.
3. Fit all models on the remaining 85%. Never fit vocabulary/tokenizer learned statistics using evaluation rows. (A pretrained tokenizer can be loaded as an external artifact.)
4. Choose hyperparameters and the two probability thresholds using **only the internal tuning split**. If comparing models, freeze the choice before public evaluation.
5. For the controlled ablation, evaluate the exact same fixed CodeBERT checkpoint at thresholds 0.50 and at tuned thresholds. Otherwise a new training run may confound the threshold comparison.
6. Run the public evaluation only once for each predeclared final condition and report each condition, not only the best. Do not optimize against public scores.
7. Hidden organizer test scores are unavailable until submission and must not be invented.

**Metrics:** For each of the 2 labels calculate TP/FP/TN/FN, precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2PR/(P+R), plus accuracy. Primary predictive value is `(F1_weakness + F1_MAT)/2`. The full organizer ranking includes mean F1 (60%), time (20%), GFLOPs (20%). Do not invent a normalized competition score without its exact notebook implementation.

## 5. Experiments — step-by-step execution

### E0 — Validate environment and audit data

- Create a Python virtual environment, `pip install -r requirements.txt`.
- `python -m pytest -q`.
- `python -m src.audit_dataset` (internet connection required).
- Copy audit results into results table; confirm `Function`, `Label` schema and that each label is binary.
- Capture `python --version`, `pip freeze`, CPU/GPU, GPU memory, OS, dataset revision and wall-clock start/end times.

### E1 — Lightweight ML control

- Run `python -m src.train_tfidf`.
- The script builds TF-IDF text representations and fits a shared-feature, two-output baseline.
- Report internal threshold selection, task-level F1, and time.
- After freezing procedure, run with `--evaluate-public-test` for the public evaluation benchmark.
- Caveat: separate label classifiers under a shared vectorizer are an inexpensive reference, not the organizer's shared neural multitask architecture.

### E2 — CodeBERT controlled reference

- Run `python -m src.train_transformer --epochs 2 --batch-size 8 --max-length 256` on a GPU if available.
- Train once and save the model checkpoint and tokenizer.
- To compare against published CodeBERT, identify differences in epoch count, sampling, optimizer, architecture head, input length, training budget and threshold. These matter: a result from this code is **not automatically a reproduction of the official baseline**.
- Complete `results/codebert_tuning.json`.
- Preserve trained checkpoint and environment manifest.

### E3 — Threshold ablation (essential)

- Using the **same saved model and same predictions**, compare default `[0.50,0.50]` and two thresholds selected on the internal tuning split.
- Predetermine threshold grid 0.10–0.90 in steps of 0.05; do not choose based on public evaluation outcomes.
- Compare MAT recall, MAT precision and F1, weakness F1 and overall mean F1.
- Report both positive and negative effects. A higher recall can reduce precision and may not increase F1.

### E4 — Sequence-length ablation (conditional on computing time)

- Train/evaluate matched configurations with 128 and 256 tokens; add 512 only if practical.
- Keep data split, seed, steps/epochs, batch-size policy, hardware and tuning policy explicit. If maximum length forces lower batch size, record it as a confound.
- For each input cap report average retained/truncated tokens, accuracy/F1, wall-clock training time and inference throughput.
- If you cannot actually run this, label it **proposed future work**, not a completed experiment.

### E5 — Qualitative error analysis

- From frozen public predictions, identify false positives and false negatives for both labels.
- Sample up to 10 of each error type with random seed. Do not expose sensitive repository data or copy long code functions into paper.
- Two researchers independently review a small sample using a rubric: ambiguous context, missing surrounding function context, noisy labels, complex control flow, comment dependence, or uncertain.
- Resolve differences by discussion; report counts of categories and illustrative paraphrased cases. Do not claim human agreement statistics unless calculated.

## 6. Results tables — fill only from computed outputs

**Table A: published organizer reference (not measured by this team).**

| Model | Weakness F1 | MAT F1 | Mean F1 | Source |
|---|---:|---:|---:|---|
| Organizer CodeBERT | 0.939 | 0.881 | 0.910 | Official organizer GitHub README |

**Table B: our measured public results.**

| Model / condition | Weakness P | Weakness R | Weakness F1 | MAT P | MAT R | MAT F1 | Mean F1 | ms/record |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| CodeBERT, 0.50 thresholds | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| CodeBERT, tuned thresholds | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| CodeBERT, 128 tokens (if run) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| CodeBERT, 512 tokens (if run) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

**Table C: ablation details.**

| Condition | Internal weakness threshold | Internal MAT threshold | Δ mean F1 vs our 0.50 checkpoint | Δ runtime | Notes |
|---|---:|---:|---:|---:|---|
| Default | 0.50 | 0.50 | 0 | Reference | Same checkpoint |
| Tuned | TBD | TBD | TBD | TBD | Same checkpoint |

A full paper should also report N and prevalence, full configuration, confusion matrices, and at least 1-2 representative error categories.

## 7. Fully specified paper outline (course version: 5–7 pages before bibliography)

### Title and authors
*Can Simple Tuning Improve Multitask Code Classification? An Empirical Study of CodeBERT on Weakness and Technical-Debt Labels*; Grant Hernandez and Ryley Wickstrom, Colorado State University.

### Abstract (150–200 words)
State the maintenance problem; two-label classification task; organizer dataset and published model; your comparisons (TF-IDF, CodeBERT, tuned thresholds, sequence length if performed); exact evaluated F1 results (insert after running); speed tradeoff and one-sentence takeaway. **Do not include numeric claims about your model until run.**

### 1. Introduction (about 550 words)
Paragraph 1: why insecure code and unaddressed debt are expensive to discover manually.  
Paragraph 2: source functions and natural-language comments jointly contain relevant evidence.  
Paragraph 3: CodeBERT multitask baseline, published F1 and compute tradeoffs.  
Paragraph 4: research gap: whether straightforward decisions (thresholds, length, simpler model) materially help.  
Paragraph 5: four RQs and three contributions: controlled evaluation, reproducible scripts, error analysis.

### 2. Related Work (about 500 words)
- Feng et al., 2020, CodeBERT: pretrained code-and-language representations.
- Mock et al., 2024, MADE-WIC: multiple annotations of code weakness.
- Russo et al., 2025, multitask SATD + vulnerability detection.
- 2027 organizers: official CodeBERT architecture and scoring constraints.
- Synthesis: our work tests inexpensive tuning rather than inventing a new encoder. Distinguish prior findings from your proposed hypotheses.

### 3. Method (about 900 words)
3.1 Dataset: origin, schema, counts, availability, original source repositories.  
3.2 Design: internal tuning held out from public train; public eval reserved for frozen runs; seed 42.  
3.3 Models: TF-IDF, shared CodeBERT encoder/two-output head.  
3.4 Training: hyperparameters and environment exact version manifest.  
3.5 Experiments: threshold grid and same-checkpoint comparison; length ablation.  
3.6 Metrics and efficiency: task F1/precision/recall, macro two-task average, time/memory/GFLOPs caveat.  
3.7 Integrity: duplicate/overlap audit, leakage controls and documentation.

### 4. Results (about 650 words, write AFTER execution)
4.1 Data audit: class balances, duplicates, project coverage, program lengths.  
4.2 Table B: quantitative model comparison. Say whether RQ1 supported with exact deltas.  
4.3 Table C: paired threshold comparison and effect on precision vs recall.  
4.4 Length/time: if run, include a speed-vs-F1 figure and hardware.  
4.5 Error analysis: counts and selected anonymized/paraphrased examples.  
**Rule:** No imaginary results or fake p-values.

### 5. Discussion (about 550 words)
Explain rather than repeat results: Which label improved? Were changes practically meaningful? How much compute is required? What does a false negative mean for security? Discuss baseline comparison fairness, project-level leakage risk, annotation noise and whether findings may transfer to new repositories. Be explicit about results that contradict expectations.

### 6. Threats to Validity (about 350 words)
Construct: binary labels do not fully capture severity or debt.  
Internal: hyperparameter search on small held-out data; run-to-run randomness.  
External: MADE-WIC projects may not represent proprietary systems or new languages.  
Conclusion: small changes may fall within seed variability; time comparisons depend on hardware.  
Mitigations: preset grid, fixed random seed, multiple seeds if possible, report exact package versions and complete runs.

### 7. Conclusion and Future Work (about 250 words)
Summarize only observed RQ answers. Future work: project-disjoint evaluation, class-weighted loss, longer contexts, model distillation and independent human review. If no gain, give a defensible negative-result conclusion.

### References
Use real citations with DOI/URLs from `docs/references_verified.md`, and cite each technical claim in text. Do not treat planned 2027 proceedings as a published peer-reviewed paper without confirming their status.

**Conference version:** NLBSE competition calls for 2–4 pages including references, while the class explicitly requires 4 pages or more and specific sections. Produce distinct course and conference versions rather than trying to squeeze a long course report into the competition format.

## 8. Figures and presentation

- **Figure 1:** `Function` text -> CodeBERT tokenizer -> shared encoder -> `[weakness logit, MAT logit]` -> sigmoid -> thresholds -> labels.
- **Figure 2:** grouped bars per task for actual measured F1 by condition.
- **Figure 3:** scatter plot of mean F1 versus inference time for real measurements.
- **Presentation slide 1 (Grant):** problem, questions, published baseline and split.
- **Slide 2 (Ryley):** data and model architecture.
- **Slide 3 (Grant):** experiments, validation, metrics.
- **Slide 4 (Ryley):** actual results, error analysis, limitations.
- **Slide 5 (both):** takeaway and run instructions, live code walkthrough.
- Both students present and explain code functions; use course's exact in-person or online submission rules. Online video up to 3 minutes, private, faces shown, link in designated `.txt` file.

## 9. Team split and handoff

**Grant:** repository/README, audit checks, TF-IDF control, evaluation metrics, visuals, Results and Discussion.  
**Ryley:** organizer notebook comparison, CodeBERT training and checkpoint, hyperparameter tracking, Related Work and Method.  
**Both:** threshold ablation, code review, final paper, troubleshooting, presentation and peer evaluation.

A useful Git workflow: main branch stays working; `grant/evaluation` and `ryley/codebert` branches, pull requests reviewed by the other teammate, tags `v0.1-data-audit`, `v0.2-baselines`, `v1.0-submission`.

## 10. Concrete execution order

1. Obtain instructor approval for revised classification topic and update the Groups spreadsheet and Discord `teams` channel with names, team and description.
2. Create a private working GitHub repo (publicize only as required for competition) and replace README's placeholder repo URL.
3. Install dependencies; run and capture unit test output.
4. Download data, run audit; save hash of dataset revision, environment and audit logs.
5. Run TF-IDF as early sanity baseline.
6. Train CodeBERT with fixed seed, freeze checkpoint.
7. Generate validation probabilities and compare default-vs-tuned threshold at identical checkpoint.
8. Run one or two additional length configs if resources permit.
9. Freeze choices; evaluate on participant public evaluation set.
10. Generate result tables and figures from actual files.
11. Analyze error samples; write RQ-by-RQ results and honest limitations.
12. Finalize paper (with required Abstract/Introduction/Related Work/Method/Results/Discussion/Conclusion/Future Work).
13. Export paper to conference style and course style as separate deliverables.
14. Record presentation and code demo or prepare live presentation; both teammates participate.
15. Build `group_name.zip` with code, README, requirements, model/checkpoint or documented model download, tests, scripts and paper. Check ZIP from a clean environment.
16. Submit peer-evaluation spreadsheet privately in Teams, copying instructor and TAs; do not include peer grades in shared repo.
17. If submitting to venue for extra credit: obtain approval and publish appropriate replication package to Zenodo or Figshare, respecting any anonymity rule.

## 11. Acceptance checklist

- [ ] Instructor approves change from generation to classification
- [ ] Spreadsheet and Discord updated
- [ ] Tests pass; full data pipeline exercised
- [ ] Dataset audit and version recorded
- [ ] Baseline comparison is fair and correctly labeled
- [ ] Threshold ablation compares one checkpoint
- [ ] Public evaluation metrics actually measured and preserved
- [ ] No fabricated results, citations or runtime numbers
- [ ] Complete report and citations
- [ ] README includes real repo URL and copy-paste run commands
- [ ] ZIP includes everything needed for reproduction
- [ ] Every teammate explains code in presentation
- [ ] Private peer evaluation submitted separately

## 12. Key source links

- Official rules: https://nlbse2027.github.io/tools/
- Organizer code, data schema and metrics: https://github.com/nlbse2027/code-comment-classification
- Official dataset: https://huggingface.co/datasets/NLBSE/nlbse27-code-comment-classification
- Official baseline model: https://huggingface.co/nlbse2027
- CodeBERT paper: https://aclanthology.org/2020.findings-emnlp.139/
- MADE-WIC DOI: https://doi.org/10.1145/3691620.3695348
- Multitask study DOI: https://doi.org/10.1109/ICPC66645.2025.00017
