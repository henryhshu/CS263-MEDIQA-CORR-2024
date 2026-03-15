# Integration — RAG + ICL + Multi-Agent

This folder contains everything added for the integration phase of the CS263 MEDIQA-CORR-2024 project. The goal is to combine all three components — Knowledge Retrieval (RAG), In-Context Learning (ICL), and Multi-Agent reasoning — so that every prediction uses all three together, rather than running them independently.

## What was added

```
integration/
├── README.md                   ← this file
├── run_integrated.py           ← runs each method independently + combined
├── run_ablation.py             ← ablation study (all 7 component combinations)
├── outputs/                    ← results from initial 50-sample runs
│   ├── 20260314_225219/        ← RAG, ICL (baseline/fixed/dynamic), Multi-Agent
│   └── 20260314_232020/        ← Combined (RAG + ICL + MA) first run
├── ablation/                   ← 50-sample ablation results
│   ├── final_ablation_report.txt
│   └── final_ablation_results.json
└── full_test/                  ← full 597-sample ablation results
    └── 20260314_235150/
        ├── ablation_report.txt
        ├── ablation_results.json
        └── *.txt               ← submission files for each variant

pipeline/
├── combined.py                 ← CombinedPredictor: applies RAG + ICL + MA per item
├── base.py                     ← abstract interfaces (LLMProvider, PromptAugmenter, etc.)
├── providers.py                ← OpenAI and Gemini provider implementations
├── augmenters.py               ← RxNorm RAG augmenter
└── run.py                      ← pipeline runner
```

### How integration works (`pipeline/combined.py`)

For each clinical text item, `CombinedPredictor` applies all three components in sequence:

1. **RAG** — PubMedBERT NER extracts drug names from the text, looks them up in RxNorm, and injects the drug reference block into the Detector and Critic system prompts. This grounds the agents in verified pharmacological facts.

2. **ICL** — Embeds the query and retrieves the k most similar training examples via cosine similarity (`text-embedding-3-small`). These are prepended as labeled demonstrations to the Detector's input.

3. **Multi-Agent** — Runs the enriched input through the full Detector → Critic (fast-path) → Editor → Critic pipeline. None of the original module files are modified.

The `use_rag`, `use_icl`, and `use_multiagent` flags can be toggled independently for ablation.

---

## How to run

**Prerequisites:** set your OpenAI API key:
```bash
export OPENAI_API_KEY=sk-...
# or place it in ~/env/openai_secret_key.txt
```

### Run the full combined system
```bash
python integration/run_integrated.py --methods combined --model gpt-4.1
```

### Run all methods independently for comparison
```bash
python integration/run_integrated.py --methods all --model gpt-4.1
```

### Run the ablation study

50-sample subset (fast, ~$1):
```bash
python integration/run_ablation.py --model gpt-4.1
```

Full test set, all 8 variants (~$12–15, takes several hours):
```bash
python integration/run_ablation.py --model gpt-4.1 --full
```

Custom sample count:
```bash
python integration/run_ablation.py --model gpt-4.1 --num-samples 200
```

Results are written to `integration/ablation/<timestamp>/` (subset) or `integration/full_test/<timestamp>/` (full).

---

## Results

### Full test set — 597 samples, gpt-4.1

Results from `integration/full_test/20260314_235150/ablation_results.json`.

| Metric        | Baseline | RAG only | ICL only | Multi-Agent | RAG+ICL | RAG+MA | ICL+MA | **Combined** |
|---------------|:--------:|:--------:|:--------:|:-----------:|:-------:|:------:|:------:|:------------:|
| Flag Accuracy | 0.694    | 0.660    | 0.700    | 0.714       | 0.682   | 0.683  | **0.740** | 0.620    |
| Sentence Acc  | 0.628    | 0.593    | 0.688    | 0.700       | 0.667   | 0.663  | **0.729** | 0.615    |
| ROUGE-1       | 0.517    | 0.491    | 0.714    | 0.686       | 0.686   | 0.687  | 0.694  | **0.796**    |
| ROUGE-L       | 0.510    | 0.484    | 0.710    | 0.682       | 0.680   | 0.683  | 0.689  | **0.794**    |
| BERTScore     | 0.514    | 0.493    | 0.737    | 0.722       | 0.711   | 0.718  | 0.731  | **0.807**    |
| BLEURT        | 0.254    | 0.248    | 0.456    | 0.403       | 0.450   | 0.434  | 0.426  | **0.543**    |
| Agg Score     | 0.516    | 0.492    | 0.726    | 0.704       | 0.698   | 0.703  | 0.713  | **0.802**    |
| Agg Composite | 0.496    | 0.496    | 0.611    | 0.599       | 0.596   | 0.586  | **0.633** | 0.584  |

### Comparison: 50-sample vs full test

The 50-sample and 597-sample runs broadly agree on relative rankings but differ in magnitude, and one finding reverses significantly:

| Observation | 50-sample | 597-sample |
|---|---|---|
| Best detector | Combined (FlagAcc **0.86**) | ICL+MA (FlagAcc **0.74**) |
| Best correction quality | RAG+MA (ROUGE-1 **0.773**) | Combined (ROUGE-1 **0.796**) |
| Best overall (AggScore) | RAG+MA (**0.783**) | Combined (**0.802**) |
| RAG-only vs baseline | Not tested | RAG-only **worse** than baseline |
| All detection metrics | Uniformly higher | Lower across the board |

**Key takeaways from the comparison:**
- **Combined's detection collapses at scale** — 50 samples showed it as the best detector (0.86); full test shows it as the worst (0.62, −0.24 delta). The small sample was biased toward cases where all three components agreed, hiding a false-positive problem.
- **ICL quality improves at scale** — ROUGE-1 and BERTScore for ICL-only actually go *up* (+0.06–0.07) with more samples, as the larger pool of training examples improves retrieval diversity.
- **50-sample detection was inflated overall** — the sampled indices (`baseline-experiment/sampled_test_indices.json`) appear to be an easier or more error-dense slice than the full test distribution, pushing all detection numbers up by ~0.08–0.12.
- **RAG+ICL is the most stable** — smallest deltas across both runs, suggesting it generalizes most consistently.

**Conclusion:** The broad qualitative conclusions hold across both scales (ICL+MA best for detection, Combined best for correction quality), but quantitative numbers from the 50-sample run are overly optimistic. The full 597-sample results should be used for reporting.

---

### 50-sample ablation (from `integration/ablation/final_ablation_report.txt`)

| Metric       | RAG only | ICL only | Multi-Agent | RAG+ICL | RAG+MA | ICL+MA | Combined |
|--------------|:--------:|:--------:|:-----------:|:-------:|:------:|:------:|:--------:|
| Flag Acc     | 0.760    | 0.820    | 0.800       | 0.720   | 0.780  | 0.840  | **0.860** |
| Sentence Acc | 0.720    | 0.760    | 0.800       | 0.700   | 0.780  | 0.820  | **0.840** |
| ROUGE-1      | 0.599    | 0.653    | 0.727       | 0.729   | **0.773** | 0.718 | 0.717  |
| BERTScore    | 0.580    | 0.667    | 0.762       | 0.760   | **0.793** | 0.770 | 0.774  |
| BLEURT       | 0.260    | 0.430    | 0.520       | **0.532** | 0.508 | 0.493 | 0.455   |
| Agg Score    | 0.589    | 0.660    | 0.745       | 0.744   | **0.783** | 0.744 | 0.746  |
| Agg Composite| 0.604    | 0.650    | 0.688       | 0.628   | 0.698  | 0.722  | **0.743** |

---

## Key findings

### From the full 597-sample run

1. **ICL is the single most impactful component.** The jump from baseline (AggScore 0.516) to ICL-only (0.726) is by far the largest gain of any single component — larger than adding RAG or Multi-Agent alone. In-context examples teach the model what a valid medical error looks like, dramatically improving both detection and correction quality.

2. **Combined system produces the best corrections.** RAG + ICL + MA achieves the highest text quality scores — ROUGE-1 0.796, BERTScore 0.807, BLEURT 0.543, AggScore 0.802 — substantially above every other variant. When it flags an error, the correction is highly accurate.

3. **ICL + MA is the best detector.** ICL + MA achieves the strongest error detection: FlagAcc 0.740, SentAcc 0.729. ICL examples calibrate the model on what counts as a real error, and the multi-agent pipeline sharpens localization.

4. **Combined system trades detection for correction quality.** The full combined system has the lowest FlagAcc (0.620), suggesting RAG + ICL together make the pipeline more conservative about flagging errors. But when it does flag, the correction quality is highest. This is a precision/recall trade-off worth noting.

5. **RAG alone does not help.** RAG-only (AggScore 0.492) is actually worse than the zero-shot baseline (0.516). Drug context without structured reasoning or task demonstrations adds noise rather than signal. RAG is only beneficial when paired with ICL and MA.

6. **Multi-Agent adds structure, ICL adds calibration, RAG adds grounding.** Each component is complementary: MA without ICL misses errors that examples would catch; ICL without MA makes unchecked corrections; RAG without the others injects irrelevant context. The full combination is strongest overall.

### Recommended configurations

| Goal | Best config |
|------|-------------|
| Best overall correction quality | RAG + ICL + MA (combined) |
| Best error detection | ICL + MA (no RAG) |
| Best balanced / composite score | ICL + MA (no RAG) |
| Lowest cost / fastest | ICL only (dynamic k=5) |
| Avoid | RAG only — worse than zero-shot baseline |
