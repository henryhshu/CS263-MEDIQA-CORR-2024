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
| GPT-5 | 0.66 | 0.5 | 0.5310 | 0.5700 | 0.6149 | 0.4289 | 0.5720 |
| GPT-4.1 | 0.76 | 0.74 | 0.6330 | 0.6059 | 0.6959 | 0.6038 | 0.6449 |
| Gemini 2.5 Flash | 0.74 | 0.66 | 0.6336 | 0.6589 | 0.6752 | 0.5542 | 0.6559 |

*Note: Gemini 2.5 Flash BLEURT score not computed due to package compatibility. AggregateComposite and AggregateScore are based on ROUGE1 + BERTSCORE only (2 metrics instead of 3).

