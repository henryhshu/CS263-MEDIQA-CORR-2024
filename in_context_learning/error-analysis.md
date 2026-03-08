# In-Context Learning Error Analysis

This report compares the sampled 50-example test subset across baseline, fixed few-shot ICL, and dynamic retrieval-based ICL with multiple `k` values.

## Behavioral Summary

| Method | False Positives | False Negatives | Wrong Sentence on Error | Exact Triplet Matches | Avg Pred Correction Tokens | Long Corrections | Explanatory Corrections |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline GPT-4.1 | 6 | 6 | 1 | 21 | 22.2 | 10 | 11 |
| Fixed ICL GPT-4.1 | 6 | 4 | 2 | 22 | 14.8 | 3 | 4 |
| Dynamic ICL k=1 | 5 | 4 | 4 | 22 | 20.4 | 7 | 13 |
| Dynamic ICL k=3 | 12 | 4 | 3 | 18 | 16.0 | 9 | 8 |
| Dynamic ICL k=5 | 5 | 3 | 2 | 23 | 14.3 | 3 | 4 |
| Dynamic ICL k=8 | 7 | 5 | 1 | 22 | 14.2 | 3 | 5 |

## Main Patterns

- Fixed few-shot mainly helps by reducing misses and over-explanation at the same time: false negatives drop from `6` to `4`, while average predicted correction length falls from `22.2` to `14.8` tokens.
- Dynamic `k=5` is the best retrieval setting for task accuracy on this subset. It has the fewest false negatives (`3`), ties for the fewest false positives (`5`), and produces the most exact full matches (`23`).
- Dynamic `k=3` is the noisiest setting. Its false positives jump to `12`, which explains why its ROUGE/BERTScore/BLEURT can look strong while its error-flag and sentence-detection accuracy fall sharply.
- Dynamic `k=8` pushes text-similarity metrics up, but it gives back some extraction accuracy compared with `k=5`: false positives rise from `5` to `7` and false negatives rise from `3` to `5`.
- Dynamic `k=1` is comparatively conservative. It stays close to `k=5` on false-positive/false-negative counts, but it misses more sentence locations (`4` vs `2`), so its overall composite score remains lower.

## Official Main Comparison

| Model | Error Flag Accuracy | Error Sentence Detection Accuracy | ROUGE1 | BERTSCORE | BLEURT | AggregateComposite | AggregateScore |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline GPT-4.1 | 0.76 | 0.74 | 0.633 | 0.6059 | 0.6959 | 0.6038 | 0.6449 |
| Fixed ICL GPT-4.1 | 0.8 | 0.76 | 0.6408 | 0.6565 | 0.6861 | 0.6373 | 0.6611 |
| Dynamic ICL GPT-4.1 (k=5) | 0.84 | 0.8 | 0.6664 | 0.681 | 0.6845 | 0.6787 | 0.6773 |

## Official Dynamic k Sweep

| Model | Error Flag Accuracy | Error Sentence Detection Accuracy | ROUGE1 | BERTSCORE | BLEURT | AggregateComposite | AggregateScore |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dynamic ICL k=1 | 0.82 | 0.74 | 0.5752 | 0.6004 | 0.6523 | 0.6325 | 0.6093 |
| Dynamic ICL k=3 | 0.68 | 0.62 | 0.6813 | 0.6969 | 0.705 | 0.5333 | 0.6944 |
| Dynamic ICL k=5 | 0.84 | 0.8 | 0.6664 | 0.681 | 0.6845 | 0.6787 | 0.6773 |
| Dynamic ICL k=8 | 0.76 | 0.74 | 0.739 | 0.7663 | 0.7367 | 0.6438 | 0.7473 |

## Representative Examples

### Fixed Few-Shot Fixes a Baseline Miss
Text ID: `ms-test-43`

Text excerpt:
```
0 | A 30-year-old patient comes to the emergency room with a chief complaint of left chest pain and a productive cough with purulent sputum for 1 week.
1 | He also complains of shortness of breath.
2 | He said he had been previously diagnosed with influenza but did not follow the doctor's instructions.
3 | His vitals include: heart rate 70/min, respiratory rate 22/min, temperature 38.7 C (101.7 F), blood pressure 120/60 mm Hg, and SO2 80%.
...
```

Gold: flag=1, sentence=4, correction=The differential diagnosis includes pneumonia.
Baseline: flag=1, sentence=6, correction=Hemoglobin 14 g/dL
Fixed: flag=1, sentence=4, correction=The differential diagnosis includes secondary bacterial pneumonia (e.g., Streptococcus pneumoniae, Staphylococcus aureus).
Dynamic k=5: flag=1, sentence=4, correction=The differential diagnosis includes secondary bacterial pneumonia (such as Staphylococcus aureus or Streptococcus pneumoniae).

### Dynamic k=5 Fixes a Fixed-ICL Error
Text ID: `ms-test-67`

Text excerpt:
```
0 | A 74-year-old woman with a history of hypertension, hyperlipidemia, and type 2 diabetes mellitus is brought to her primary care doctor by her son who is concerned about the patient's ability to manage her activities of daily living.
1 | She has been living alone since her husband passed away 2 years ago.
2 | She has remained an active member of her retirement community's wine club; however, she stopped attending the meetings 3 weeks ago.
3 | She is generally able to her maintain her house, but her house appears more disorganized over the past 2 weeks.
...
```

Gold: flag=1, sentence=15, correction=The patient is diagnosed with delirium after she can recall 0/3 words after 5 minutes.
Fixed: flag=1, sentence=16, correction=She has a low-frequency tremor in her right arm that decreases in severity with deliberate movement.
Dynamic k=1: flag=1, sentence=16, correction=She has a low-frequency tremor in her right arm that increases in severity with deliberate movement (intention tremor), which is more characteristic of cerebellar dysfunction rather than Alzheimer dementia.
Dynamic k=5: flag=1, sentence=15, correction=The patient is diagnosed with delirium due to acute cognitive changes, inattention, altered level of consciousness, and fluctuating symptoms, rather than Alzheimer dementia.

### Larger k Hurts Extraction Accuracy
Text ID: `ms-test-207`

Text excerpt:
```
0 | A 55-year-old woman comes to the office complaining of a dry mouth for the past few months.
1 | She has to drink water more frequently, as she finds it difficult to chew and swallow solid foods.
2 | She has to wake up 3-5 times each night to drink due to intense thirst.
3 | She also complains of a foreign body sensation in both the eyes for the past month.
...
```

Gold: flag=0, sentence=-1, correction=NA
Dynamic k=5: flag=0, sentence=-1, correction=NA
Dynamic k=8: flag=1, sentence=14, correction=Leukocytes 4,000-10,000/mm3

