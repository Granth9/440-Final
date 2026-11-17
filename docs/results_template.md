# Results tables and exact claim wording

**Never fill a cell without evidence from actual experiment output.**

## Dataset audit
| Audit measure | Train | Public evaluation |
|---|---:|---:|
| Records | TBD | TBD |
| Weakness-positive % | TBD | TBD |
| SATD-positive % | TBD | TBD |
| `[0,0]` / `[1,0]` / `[0,1]` / `[1,1]` counts | TBD | TBD |
| Duplicate function rows | TBD | TBD |

Exact function hashes appearing in both splits: **TBD**. Shared project names: **TBD**.

## Performance
| Model / experiment | Weakness P | Weakness R | Weakness F1 | SATD P | SATD R | SATD F1 | Mean F1 | Inference sec/item |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Published CodeBERT baseline | .943 | .935 | .939 | .978 | .802 | .881 | .910 | not supplied here |
| TF-IDF control | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Reproduced CodeBERT | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Separate thresholds | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 128-token CodeBERT | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 512-token CodeBERT | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Sample explanatory sentences to complete after evaluation
- "On the public evaluation split, our final model achieved weakness F1 = [X] and SATD F1 = [Y], giving mean F1 = [Z]."
- "Compared with the published CodeBERT baseline mean F1 of 0.910, the difference was [Z - .910]. This is a published-reference comparison, not necessarily a controlled replication."
- "On identical hardware and the same example order, the inference time changed from [A] to [B] seconds/example."
- "The most frequent error category was [observed category], accounting for [observed count] of [sample size] reviewed errors."
- "We could not establish that [intervention] caused the difference because [specific limitation]."
