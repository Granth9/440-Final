# Team plan, division of labor and timeline

## Suggested division (both members should review the entire project)
**Grant Hernandez:** set up repository/environment; implement/check lightweight TF-IDF comparison; evaluate metrics, plots and runtime; report Results/Discussion.

**Ryley Wickstrom:** inspect organizer baseline/notebook; train/tune CodeBERT on GPU; analyze errors and papers; draft Introduction/Related Work/Method.

**Both:** approve topic, audit leakage/fairness, review all evidence, merge work, rehearse presentation, show code functions, contribute to peer evaluation. These are suggested roles, not claims about work already completed.

## Dates (based on official NLBSE schedule; course deadlines may differ)
- **Nov 14:** obtain instructor approval for corrected topic; create repo; run smoke tests; verify HF schema.
- **Nov 15–17:** reproduce baseline/perform E1; start GPU E2; write introduction/methods.
- **Nov 18–20:** tune on internal validation; implement model checkpoint/evaluation adapter; review errors.
- **Nov 21–23:** freeze choices, compute public-test scores; finish results and discussion; prepare graphs.
- **Nov 24–26:** format venue paper, verify notebooks/README, video or live slides, peer evaluation, make zip, check collaborators.
- **Nov 27, 2026 (AoE):** optional external NLBSE tool competition submission (must meet venue and instructor criteria).

## Three-minute presentation outline (for online students)
- 0:00–0:30 — Problem, official NLBSE task, two output labels (Grant).
- 0:30–1:00 — Data and baseline; methodology and leakage prevention (Ryley).
- 1:00–1:45 — Walk through loading, feature/model training, threshold choosing (Grant).
- 1:45–2:25 — Model results, F1 and runtime, one actual error example (Ryley).
- 2:25–3:00 — Limitations, conclusion, GitHub reproduction instructions (both).

Record with faces visible while presenting, professional background and private link/file; the assignment explicitly says missing video earns zero for online students.
