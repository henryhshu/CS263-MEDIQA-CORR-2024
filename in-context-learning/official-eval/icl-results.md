# In-Context Learning Results

The evaluation results are based on a random sample of 50 data samples to reduce API usage costs.
| Model | Error Flag Accuracy | Error Sentence Detection Accuracy | ROUGE1 | BERTSCORE | BLEURT | AggregateComposite | AggregateScore |
| ----- | --- | --- | --- | --- | --- | --- | --- |
| Baseline GPT-4.1 | 0.76 | 0.74 | 0.633 | 0.6059 | 0.6959 | 0.6038 | 0.6449 |
| Fixed ICL GPT-4.1 | 0.8 | 0.76 | 0.6408 | 0.6565 | 0.6861 | 0.6373 | 0.6611 |
| Dynamic ICL GPT-4.1 | 0.68 | 0.62 | 0.6824 | 0.6922 | 0.732 | 0.5371 | 0.7022 |
