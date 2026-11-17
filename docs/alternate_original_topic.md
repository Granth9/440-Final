# If the instructor wants the ORIGINAL comment generation idea

**Topic:** Improving AI-Generated Code Comment Quality Through Prompt Engineering

**Research question:** Do structured prompts that include function purpose, inputs, outputs, edge cases, and a warning not to invent behavior produce more accurate/useful code comments than a minimal baseline prompt?

**Crucial:** This is a proposed independent NLBSE-style short research paper; it is **not** the official NLBSE 2027 tool classification competition.

## Proposed method
- Build a 100-function evaluation set of open-source functions, record licenses, repository snapshots, and provenance. Avoid leakage from training/LLM example prompts.
- Compare 3 prompts with the **same** model and decoding parameters: (A) `Explain this function in one sentence.` (B) structured prompt asking for objective, inputs, outputs, notable side effects; (C) structured prompt plus factuality check and reject ungrounded claims.
- Make no claim that a model knows what functions do without inspection. Human raters independently score 0–2 on factual accuracy, relevance, clarity and hallucinated behavior. Randomize presentation and blind raters to condition.
- Report means, paired distributions, inter-rater agreement (Cohen's kappa or weighted kappa as suitable), and real examples. Keep prompts, model identifier, outputs, annotations, and scoring code in replication package.
- Do not compare generated comments to the official 2027 CodeBERT classification baseline: these solve different tasks.

## Suggested draft title
**Does Structured Prompting Improve the Factual Quality of LLM-Generated Source Code Comments?**

## Caution
This alternative would require a **different** implementation and evaluation dataset. The accompanying `src` files are for the official classification task; do not pretend those files test generation.
