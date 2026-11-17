# Experiments and analysis sheet

## Experimental design
| ID | Treatment | Where to tune | Primary metrics | Status |
|---|---|---|---|---|
| E0 | Official published CodeBERT baseline | Published by organizers | weakness F1=.939; MAT F1=.881; mean=.910 | External published numbers |
| E1 | TF-IDF character 3–5-grams + 2 logistic heads | Internal train validation only | F1, P/R, seconds/example | Code included; not run on official dataset |
| E2 | 2-output CodeBERT, 256 tokens, 2 epochs | Internal train validation only | F1, P/R, seconds/example | Code included; not trained |
| E3 | Thresholds chosen independently per label | Internal train validation only | ΔF1, ΔP, ΔR | Implemented threshold search; not trained |
| E4 | Longer/shorter max length; optional | Internal train validation only | F1, latency, truncation rate | Proposed, not implemented |
| E5 | Imbalance-aware loss; optional | Internal train validation only | MAT P/R/F1 | Proposed, not implemented |

## Execution notes
1. First run `pytest` and smoke train. Confirm train/test sizes and label distribution match repository.
2. Implement a **reproducible exact baseline** or evaluate organizer weights using the official notebook; E0 is currently only an external reported comparator.
3. Run E1 full, then E2 full; keep package, hardware, seed, train params, compute hours, and checkpoints.
4. Choose model/thresholds on internal validation. Freeze preprocessing/hyperparameters.
5. Evaluate final frozen models on public evaluation split; do not revise them using this score.
6. For official entry, check exact output artifact and GFLOPs API in the competition notebook; it may differ from this starter.
7. Qualitatively inspect 10 FP and 10 FN per target if available, and identify patterns using anonymized snippets and licenses.

## Results table — fill with ACTUAL runs
| Method | Weakness F1 | MAT F1 | Mean F1 | Mean Δ to .910 | Inference ms/function | GFLOPs |
|---|---:|---:|---:|---:|---:|---:|
| Published organizer CodeBERT | 0.939 | 0.881 | 0.910 | 0.000 | not reported here | not reported here |
| E1 TF-IDF | TBD | TBD | TBD | TBD | TBD | TBD |
| E2 CodeBERT our run | TBD | TBD | TBD | TBD | TBD | TBD |
| E3 CodeBERT + thresholds | TBD | TBD | TBD | TBD | TBD | TBD |

## Confusion matrix (per label)
| Label | TP | FP | TN | FN | Threshold |
|---|---:|---:|---:|---:|---:|
| Weakness | TBD | TBD | TBD | TBD | TBD |
| MAT | TBD | TBD | TBD | TBD | TBD |

## Quality controls
- Never fit tokenizer/vectorizer using official public test.
- Use train-only internal validation for thresholds.
- Don't call internal validation numbers `test` results.
- If public test is evaluated multiple times during experimentation, disclose it and don't describe it as untouched.
- Track corpus license and whether excerpts can be included in publication.
