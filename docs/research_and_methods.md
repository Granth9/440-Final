# Research background and defensible methodology (verified 14 Nov 2026)

## 1. What NLBSE 2027 actually asks
The NLBSE'27 tool competition is **multitask code classification**, not code-comment generation. One model receives a formatted function containing code/comment information, and predicts **two binary values**: weakness/vulnerability and MAT (the organizer's annotation for self-admitted technical debt). The official competition website describes SATD, while the GitHub repository calls the second target `MAT`; always preserve dataset `Label[1]` and explain this terminology. Do not interchange the order. The organizers rank by mean of task F1 (60% of total), time (20%) and GFLOPs (20%). Official source: https://github.com/nlbse2027/code-comment-classification and https://nlbse2027.github.io/tools/

## 2. Actual baseline and sample sizes
The official repository reports `train`=214,586, public `test`=30,655, and hidden evaluation=61,311, derived from MADE-WIC and incorporating OSPR, Big-Vul, Devign. Its baseline is **CodeBERT** with a shared encoder and two binary outputs. Published F1 is **0.939** for weakness and **0.881** for MAT, mean **0.910**. These are organizer-reported metrics, NOT our own measurements. The public website generically describes public train/validation/hidden test while its repository calls the public holdout `test`; follow repository split names in code, and describe the public holdout as the public evaluation split in the report.

## 3. Related peer-reviewed research
- **Feng et al., 2020**, *CodeBERT: A Pre-Trained Model for Programming and Natural Languages*, Findings of EMNLP, 1536–1547. https://doi.org/10.18653/v1/2020.findings-emnlp.139  — introduces CodeBERT, a pretrained model for code and natural language.
- **Mock et al., 2024**, *MADE-WIC: Multiple Annotated Datasets for Exploring Weaknesses In Code*, ASE, 2346–2349. https://doi.org/10.1145/3691620.3695348 — fused datasets with code weaknesses and technical-debt annotations; label construction can affect comparability.
- **Russo, Melegati and Mock, 2025**, *Leveraging Multi-Task Learning to Improve the Detection of SATD and Vulnerability*, ICPC, 1–12. https://doi.org/10.1109/ICPC66645.2025.00017 and https://arxiv.org/abs/2501.15934 — investigates shared learning for both targets; reports no significant improvement over separate approaches in its tested settings. A valuable motivation for reporting negative results.
- **NLBSE 2027 organizers**, *The NLBSE'27 Tool Competition and Challenge*, forthcoming 2027; listed as competition citation on official website https://nlbse2027.github.io/tools/. **Do not falsely describe forthcoming publication as already peer-reviewed/published.**

## 4. Research questions and hypotheses
- RQ1: Does a modified CodeBERT model increase mean F1 compared with published 0.910? H1: threshold tuning/class weighting or better context may increase the MAT F1, which is weaker than vulnerability detection.
- RQ2: How do improvements affect precision and recall for each label? H2: a lower MAT threshold may improve recall but can reduce precision.
- RQ3: What runtime/memory tradeoffs arise from a lighter model? H3: TF-IDF will be cheaper to evaluate but may yield lower F1; this is a hypothesis, not an established outcome.

## 5. Methods
**Primary experiments:** (E0) published organizer baseline as reported; (E1) shared TF-IDF character representation and two logistic-regression output heads as a cheap comparison; (E2) pretrained CodeBERT shared backbone with two sigmoid outputs, fine-tuned using multi-label binary cross-entropy; (E3) tune output thresholds on an internal held-out set. Optional ablations require changes to code: length 128 vs 256 vs 512, class-weighting, model architecture or code-vs-comment features. Do not claim these ablations were run unless outputs exist.

**Data:** use official training split only for fitting and for making an internal ~85/15 stratified tuning split. Joint four-class stratification uses the two-bit labels. Since public `test` is labeled, it should still be sequestered from design/tuning; report it once after decisions are frozen. The hidden test is never accessible to participants.

**Targets:** both labels are 0/1. Thresholded output `[weakness, MAT]` strictly. For CodeBERT, `problem_type='multi_label_classification'` triggers independent binary losses rather than a four-class softmax. This enables `[1,1]`.

**Metrics:** report four metrics per label plus unweighted mean F1, with TP/FP/FN and supports if available. Report threshold choice. Mean F1 alone can obscure tradeoffs. Include prediction seconds/example (same hardware; warmed up, repeated runs for stable comparisons), model storage, peak RAM/VRAM and organizer GFLOPs where available. The supplied code times a single prediction pass and **does not calculate GFLOPs**; use official notebook for official ranking.

**Threats to validity:** duplicates or near-duplicates across projects/splits, source-project imbalance, labels inferred from mixed annotation rules, train/evaluation leakage, limited token context, class imbalance, published baseline measured in a different environment, noisy SATD labels, and a single seed. Audit samples and use 3 independent random seeds if compute permits. Report resources and any incomplete runs.

## 6. Interpreting possible outcomes
- A higher local mean F1 than 0.910 **does not guarantee** winning because timing and GFLOPs account for 40% of ranking and published results may differ under a matched environment.
- If MAT recall rises but precision falls, examine F1 and label prevalence; don't report recall gain as an unqualified success.
- If no approach beats CodeBERT, the scientifically valid conclusion is a negative result about selected methods, not a fabricated success.

## Source links / primary verification
https://nlbse2027.github.io/tools/
https://github.com/nlbse2027/code-comment-classification
https://huggingface.co/datasets/NLBSE/nlbse27-code-comment-classification
https://aclanthology.org/2020.findings-emnlp.139/
https://doi.org/10.1145/3691620.3695348
https://doi.org/10.1109/ICPC66645.2025.00017
