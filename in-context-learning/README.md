# In-Context Learning Experiments

This directory contains a single script for three experiment modes:

- `baseline`: zero-shot prompting
- `fixed`: fixed few-shot prompting using the first `k` training examples
- `dynamic`: retrieval-based few-shot prompting using embedding nearest neighbors

## Runtime requirements

- `OPENAI_API_KEY` must be set.
- Network access is required when the script loads the public `mkieffer/MEDEC` dataset or calls OpenAI.
- If `datasets` is installed, the script will use it. Otherwise it falls back to the Hugging Face dataset server over HTTP.

## Default behavior

- Loads `train` and `test` from `mkieffer/MEDEC`
- Restricts test evaluation to the sampled 50-example subset from `baseline-experiment/sampled_test_indices.json`
- Evaluates the existing baseline run in `baseline-experiment/outputs/medec-ms_gpt4_1_results_20260205_180349.txt`
- Runs `baseline`, `fixed`, and `dynamic`
- Writes submission files to `in-context-learning/outputs/`
- Writes a summary JSON to `in-context-learning/outputs/summary.json`

## Example

```bash
export OPENAI_API_KEY=...
python3 in-context-learning/in-context-learning.py \
  --model gpt-4.1 \
  --k-shot 3 \
  --reasoning-effort low
```

## Useful flags

```bash
python3 in-context-learning/in-context-learning.py --full-test
python3 in-context-learning/in-context-learning.py --modes fixed dynamic --k-shot 5
python3 in-context-learning/in-context-learning.py --train-file path/to/train.json --test-file path/to/test.json
python3 in-context-learning/in-context-learning.py --skip-existing-baseline
```

## Reported metrics

The script computes local comparison metrics from the ground-truth labels in the test split:

- `error_flag_accuracy`
- `error_sentence_accuracy`
- `correction_exact_match_all`
- `rouge1_f1_error_subset`
- `rougeL_f1_error_subset`
- `rouge1_f1_composite`
- `rougeL_f1_composite`

These are intended for project-side comparison between `baseline`, `fixed`, and `dynamic`. They are not a full reimplementation of the official MEDIQA evaluation stack.

## Official evaluation

`official_eval.py` runs the shared evaluation workflow from `evaluation/mediqa-corr-2024-eval-on-hf-dataset.py` and writes teammate-style summary tables.

Example:

```bash
python3 in-context-learning/official_eval.py \
  --baseline-file baseline-experiment/outputs/medec-ms_gpt4_1_results_20260205_180349.txt \
  --fixed-file in-context-learning/outputs/medec_fixed_gpt-4_1_20260228_155001.txt \
  --dynamic-file in-context-learning/outputs/medec_dynamic_gpt-4_1_20260228_155509.txt
```

Generated files:

- `in-context-learning/official-eval/icl-results.md`
- `in-context-learning/official-eval/icl-results.csv`

## Result

The evaluation results are based on a random sample of 50 data samples to reduce API usage costs.

| Model | Error Flag Accuracy | Error Sentence Detection Accuracy | ROUGE1 | BERTSCORE | BLEURT | AggregateComposite | AggregateScore |
|-----|---|---|---|---|---|---|---|
| Baseline GPT-4.1 | 0.76 | 0.74 | 0.6330 | 0.6059 | 0.6959 | 0.6038 | 0.6449 |
| Fixed ICL GPT-4.1 | 0.80 | 0.76 | 0.6408 | 0.6565 | 0.6861 | 0.6373 | 0.6611 |
| Dynamic ICL GPT-4.1 (`k=5`) | 0.84 | 0.80 | 0.6664 | 0.6810 | 0.6845 | 0.6787 | 0.6773 |

`k=5` is the selected dynamic setting because it gave the best dynamic performance on the task-facing metrics and the best `AggregateComposite` among the tested `k` values.

## Dynamic K Sweep

The evaluation results are based on the same random sample of 50 data samples.

| Model | Error Flag Accuracy | Error Sentence Detection Accuracy | ROUGE1 | BERTSCORE | BLEURT | AggregateComposite | AggregateScore |
|-----|---|---|---|---|---|---|---|
| Dynamic ICL `k=1` | 0.82 | 0.74 | 0.5752 | 0.6004 | 0.6523 | 0.6325 | 0.6093 |
| Dynamic ICL `k=3` | 0.68 | 0.62 | 0.6813 | 0.6969 | 0.7050 | 0.5333 | 0.6944 |
| Dynamic ICL `k=5` | 0.84 | 0.80 | 0.6664 | 0.6810 | 0.6845 | 0.6787 | 0.6773 |
| Dynamic ICL `k=8` | 0.76 | 0.74 | 0.7390 | 0.7663 | 0.7367 | 0.6438 | 0.7473 |

Interpretation:

- `k=5` is best for task accuracy and `AggregateComposite`.
- `k=8` improves the raw text-generation metrics and `AggregateScore`, but it loses ground on the error flag and error sentence tasks.
- `k=1` is competitive, but it still trails `k=5` on the chosen composite metric.
