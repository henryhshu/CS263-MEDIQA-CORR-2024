# Baseline experiment results

## Prompt
The prompt was used exactly as described in the paper(https://aclanthology.org/2024.clinicalnlp-1.57.pdf)
```
system = """
The following is a medical narrative about a patient. You are a skilled medical doctor reviewing the clinical text. The text is either correct or contains one error.
The text has a sentence per line. Each line starts with the sentence ID, followed by a pipe character then the sentence to check. Check every sentence of the text.
If the text is correct return the following output: CORRECT. If the text has a medical error, return the sentence id of the sentence containing the error,
followed by a space, and a corrected version of the sentence.
"""
```

## Result
The evaluation results are based on a random sample of 50 data samples to reduce API usage costs.
| Model | Error Flag Accuracy | Error Sentence Detection Accuracy | ROUGE1 | BERTSCORE | BLEURT | AggregateComposite | AggregateScore |
|-----|---|---|---|---|---|---|---|
| GPT-5 | 0.66 | 0.34 | 0.5293 | 0.5669 | 0.6106 | 0.4727 | 0.5689 |
| GPT-4.1 | 0.76 | 0.56 | 0.6337 | 0.6030 | 0.6945 | 0.6032 | 0.6437 |
