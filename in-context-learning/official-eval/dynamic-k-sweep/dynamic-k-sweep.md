# In-Context Learning Results

The evaluation results are based on a random sample of 50 data samples to reduce API usage costs.
| Model | Error Flag Accuracy | Error Sentence Detection Accuracy | ROUGE1 | BERTSCORE | BLEURT | AggregateComposite | AggregateScore |
| ----- | --- | --- | --- | --- | --- | --- | --- |
| Dynamic ICL k=1 | 0.82 | 0.74 | 0.5752 | 0.6004 | 0.6523 | 0.6325 | 0.6093 |
| Dynamic ICL k=3 | 0.68 | 0.62 | 0.6813 | 0.6969 | 0.705 | 0.5333 | 0.6944 |
| Dynamic ICL k=5 | 0.84 | 0.8 | 0.6664 | 0.681 | 0.6845 | 0.6787 | 0.6773 |
