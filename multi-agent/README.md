# Multi-agent experiment results

## How to run the code?
gpt-4.1 shows the best performance compare to gpt-5, gpt-5.2
```
$ python multi-agent/multi-agent-detect-critic-edit.py --split test --detector_model gpt-4.1 --critic_model gpt-4.1 --editor_model gpt-4.1
```

## Result
The evaluation results are based on a random sample of 50 data samples to reduce API usage costs.
| Model | Error Flag Accuracy | Error Sentence Detection Accuracy | ROUGE1 | BERTSCORE | BLEURT | AggregateComposite | AggregateScore |
|-----|---|---|---|---|---|---|---|
| GPT-5.2 | 0.68 | 0.6 | 0.5860 | 0.6562 | 0.6478 | 0.5024 | 0.6300 |
| GPT-4.1 | 0.74 | 0.68 | 0.6246 | 0.7082 | 0.6693 | 0.6070 | 0.6674 |
