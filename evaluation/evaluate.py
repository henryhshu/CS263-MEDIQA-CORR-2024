#!/usr/bin/env python
# coding: utf-8
"""
MEDIQA-CORR Evaluation Script
Computes: Accuracy, ROUGE, BERTSCORE, and BLEURT
"""

import re
import pandas as pd
from rouge import Rouge
import numpy as np
import math
import string
import json
from pathlib import Path
from datasets import load_dataset

# Optional dependency: bert_score
try:
    import bert_score.score as bertscore
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False

def parse_reference_dataset(dataset):
    """Parse reference dataset from HuggingFace."""
    reference_corrections = {}
    reference_flags = {}
    reference_sent_id = {}

    for item in dataset:
        text_id = item['text_id']
        corrected_sentence = item['corrected_sentence']

        if not isinstance(corrected_sentence, str):
            if math.isnan(corrected_sentence):
                corrected_sentence = "NA"
            else:
                corrected_sentence = str(corrected_sentence).replace("\n", " ").replace("\r", " ").strip()
        
        if corrected_sentence == "NA.":
            corrected_sentence = "NA"
        
        reference_corrections[text_id] = corrected_sentence
        reference_flags[text_id] = item['error_flag']
        reference_sent_id[text_id] = item['error_sentence_id']
    
    return reference_corrections, reference_flags, reference_sent_id


def parse_run_submission_file(filepath):
    """Parse model output file."""
    file = open(filepath, 'r')
    candidate_corrections = {}
    predicted_flags = {}
    candidate_sent_id = {}
    
    lines = file.readlines()
    
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            continue
            
        if not re.fullmatch(r'[a-z0-9\-]+\s[0-9]+\s\-?[0-9]+\s.+', line):
            print("Invalid line: ", line)
            continue
        
        line = re.sub(r'\s+', ' ', line)
        items = line.split()
        text_id = items[0]
        error_flag = items[1]
        sentence_id = items[2]
        corrected_sentence = ' '.join(items[3:]).strip()

        predicted_flags[text_id] = error_flag
        candidate_sent_id[text_id] = sentence_id

        # Remove quotes
        while corrected_sentence.startswith('"') and len(corrected_sentence) > 1:
            corrected_sentence = corrected_sentence[1:]
        while corrected_sentence.endswith('"') and len(corrected_sentence) > 1:
            corrected_sentence = corrected_sentence[:-1]
                   
        if error_flag == '0':
            candidate_corrections[text_id] = "NA"
        else:
            candidate_corrections[text_id] = corrected_sentence

    return candidate_corrections, predicted_flags, candidate_sent_id


def compute_accuracy(reference_flags, reference_sent_id, predicted_flags, candidate_sent_id):
    """Compute error detection accuracy."""
    matching_flags_nb = 0
    for text_id in reference_flags:
        if text_id in predicted_flags and int(reference_flags[text_id]) == int(predicted_flags[text_id]):
            matching_flags_nb += 1
    flags_accuracy = matching_flags_nb / len(reference_flags)
    
    matching_sentence_nb = 0
    for text_id in reference_sent_id:
        if text_id in candidate_sent_id and int(candidate_sent_id[text_id]) == int(reference_sent_id[text_id]):
            matching_sentence_nb += 1
    sent_accuracy = matching_sentence_nb / len(reference_sent_id)

    return {
        "Error Flags Accuracy": flags_accuracy,
        "Error Sentence Detection Accuracy": sent_accuracy
    }


def clip(value):
    return max(0, min(1, value))


def get_nlg_eval_data(reference_corrections, candidate_corrections):
    """Prepare data for NLG metrics."""
    references = []
    predictions = []
    
    counters = {
        "total_texts": 0,
        "reference_na": 0,
        "total_system_texts": 0,
        "system_provided_na": 0,
        "system_provided_correct_na": 0,
    }
    
    for text_id in reference_corrections:
        counters["total_texts"] += 1
        reference_correction = reference_corrections[text_id]
            
        if reference_correction == "NA":
            counters["reference_na"] += 1
            
        if text_id in candidate_corrections:
            counters["total_system_texts"] += 1
            candidate = candidate_corrections[text_id]
                
            if candidate == "NA":
                counters["system_provided_na"] += 1
                
            if reference_correction == "NA" and candidate == "NA":
                counters["system_provided_correct_na"] += 1
                continue
                
            if candidate == "NA" or reference_correction == "NA":
                continue
                
            references.append(reference_correction)
            predictions.append(candidate)
    
    return references, predictions, counters


def compute_bleurt(references, predictions, model_name='Elron/bleurt-base-512', batch_size=32):
    """
    Compute BLEURT scores using the Elron/bleurt-base-512 checkpoint.
    
    This uses a standard BERT-based BLEURT model that works with modern
    transformers (no TensorFlow dependency, no bleurt-pytorch needed).
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    print(f"Loading BLEURT model ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    
    # Use MPS/GPU if available, else CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)
    print(f"  Using device: {device}")
    
    all_scores = []
    for i in range(0, len(references), batch_size):
        batch_refs = references[i:i+batch_size]
        batch_preds = predictions[i:i+batch_size]
        
        with torch.no_grad():
            inputs = tokenizer(
                batch_refs, batch_preds,
                padding=True, truncation=True, max_length=512,
                return_tensors='pt'
            ).to(device)
            scores = model(**inputs).logits.flatten().cpu().tolist()
            all_scores.extend(scores)
        
        if (i // batch_size) % 10 == 0 and i > 0:
            print(f"  Processed {i}/{len(references)} pairs...")
    
    print(f"  BLEURT computed for {len(all_scores)} pairs")
    return all_scores


def compute_nlg_metrics(references, predictions, counters, include_bleurt=True, include_bertscore=True):
    """Compute ROUGE, BERTSCORE, and optionally BLEURT."""
    results = {}
    
    # ROUGE
    rouge = Rouge() 
    rouge_scores = rouge.get_scores(predictions, references)
    
    rouge1f_scores = [s["rouge-1"]["f"] for s in rouge_scores]
    rouge2f_scores = [s["rouge-2"]["f"] for s in rouge_scores]
    rougeLf_scores = [s["rouge-l"]["f"] for s in rouge_scores]
    
    results['ROUGE1'] = np.mean(rouge1f_scores)
    results['ROUGE2'] = np.mean(rouge2f_scores)
    results['ROUGEL'] = np.mean(rougeLf_scores)
    
    # BERTSCORE
    bertscores = None
    if include_bertscore:
        if not HAS_BERTSCORE:
            print("\n  WARNING: bert_score not installed. Install with: pip install bert-score")
            print("  Skipping BERTScore computation.")
        else:
            # Fix: bert_score passes tokenizer.model_max_length to tokenizers,
            # which overflows for DeBERTa (value is ~2^63). Patch it to 512.
            import bert_score.utils as _bsu
            _orig_sent_encode = _bsu.sent_encode
            def _patched_sent_encode(tokenizer, sent):
                if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length > 100_000:
                    tokenizer.model_max_length = 512
                return _orig_sent_encode(tokenizer, sent)
            _bsu.sent_encode = _patched_sent_encode

            print("Computing BERTScore (this may take a few minutes)...")
            _, _, bertScore_F1 = bertscore(
                predictions, references, 
                model_type='microsoft/deberta-xlarge-mnli', 
                lang='en', 
                device='cpu',
                verbose=True, 
                rescale_with_baseline=True
            )

            # Restore original
            _bsu.sent_encode = _orig_sent_encode

            bertscores = np.array([clip(num) for num in bertScore_F1.numpy()])
            results['BERTSCORE'] = np.mean(bertscores)
    
    # BLEURT
    if include_bleurt:
        print("\nComputing BLEURT...")
        bleurt_scores = compute_bleurt(references, predictions)
        bleurt_scores = np.array([clip(s) for s in bleurt_scores])
        results['BLEURT'] = np.mean(bleurt_scores)
        bleurt_composite = (np.sum(bleurt_scores) + counters["system_provided_correct_na"]) / counters["total_texts"]
        results['BLEURT_Composite'] = bleurt_composite
    
    # Composite scores
    rouge1_composite = (np.sum(rouge1f_scores) + counters["system_provided_correct_na"]) / counters["total_texts"]
    results['ROUGE1_Composite'] = rouge1_composite
    
    if bertscores is not None:
        bertscore_composite = (np.sum(bertscores) + counters["system_provided_correct_na"]) / counters["total_texts"]
        results['BERTSCORE_Composite'] = bertscore_composite
        
        # Aggregate (ROUGE1 + BERTSCORE) / 2
        aggregate_per_sample = (np.array(rouge1f_scores) + bertscores) / 2
        aggregate_composite = (np.sum(aggregate_per_sample) + counters["system_provided_correct_na"]) / counters["total_texts"]
        results['AggregateComposite'] = aggregate_composite
        results['AggregateScore'] = np.mean(aggregate_per_sample)
    
    return results


def load_indices(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))["indices"]


if __name__ == "__main__":
    import argparse

    _script_dir = Path(__file__).parent
    _default_submission = str(_script_dir / ".." / "baseline-experiment" / "outputs" / "medec-ms_gemini-2.5-flash_results_20260206_192929.txt")
    _default_indices = str(_script_dir / ".." / "baseline-experiment" / "sampled_test_indices.json")

    parser = argparse.ArgumentParser(description="MEDIQA-CORR Evaluation")
    parser.add_argument(
        "--submission", "-s", type=str,
        default=_default_submission,
        help="Path to the submission file to evaluate",
    )
    parser.add_argument(
        "--indices", "-i", type=str,
        default=_default_indices,
        help="Path to sampled indices JSON (ignored if --full is set)",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Evaluate against the full test set instead of sampled indices",
    )
    parser.add_argument(
        "--no-bleurt", action="store_true",
        help="Skip BLEURT computation (faster evaluation)",
    )
    parser.add_argument(
        "--no-bertscore", action="store_true",
        help="Skip BERTScore computation (faster evaluation)",
    )
    parser.add_argument(
        "--split", type=str, default="test",
        help="Dataset split (default: test)",
    )
    args = parser.parse_args()

    submission_file = args.submission
    split = args.split

    print("=" * 60)
    print("MEDIQA-CORR Evaluation")
    print("=" * 60)
    print(f"  Submission: {submission_file}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset("mkieffer/MEDEC-MS", split=split)

    if not args.full:
        loaded_indices = load_indices(args.indices)
        dataset = dataset.select(loaded_indices)
        print(f"Loaded {len(dataset)} samples (sampled subset)")
    else:
        print(f"Loaded {len(dataset)} samples (full test set)")

    # Parse reference and predictions
    reference_corrections, reference_flags, reference_sent_id = parse_reference_dataset(dataset)
    candidate_corrections, candidate_flags, candidate_sent_id = parse_run_submission_file(submission_file)

    # Check coverage
    missing = set(reference_flags.keys()) - set(candidate_flags.keys())
    if missing:
        print(f"\n  WARNING: {len(missing)} text_ids in reference but not in submission")

    # Compute accuracy
    print("\n" + "-" * 40)
    print("ACCURACY METRICS")
    print("-" * 40)
    accuracy_results = compute_accuracy(reference_flags, reference_sent_id, candidate_flags, candidate_sent_id)
    for k, v in accuracy_results.items():
        print(f"  {k}: {v:.4f}")

    # Compute NLG metrics
    print("\n" + "-" * 40)
    print("NLG METRICS")
    print("-" * 40)
    references, predictions, counters = get_nlg_eval_data(reference_corrections, candidate_corrections)
    print(f"  Samples with corrections (for NLG eval): {len(references)}")
    print(f"  Correct NA predictions: {counters['system_provided_correct_na']}")

    include_bleurt = not args.no_bleurt
    include_bertscore = not args.no_bertscore
    nlg_results = compute_nlg_metrics(references, predictions, counters, include_bleurt=include_bleurt, include_bertscore=include_bertscore)
    print("\nResults:")
    for k, v in nlg_results.items():
        print(f"  {k}: {v:.4f}")

    # Summary table
    print("\n" + "=" * 60)
    print(f"SUMMARY — {Path(submission_file).stem}")
    print("=" * 60)
    print(f"| Metric                          | Value  |")
    print(f"|--------------------------------|--------|")
    print(f"| Error Flag Accuracy            | {accuracy_results['Error Flags Accuracy']:.4f} |")
    print(f"| Error Sentence Detection Acc   | {accuracy_results['Error Sentence Detection Accuracy']:.4f} |")
    print(f"| ROUGE1                         | {nlg_results['ROUGE1']:.4f} |")
    if 'BERTSCORE' in nlg_results:
        print(f"| BERTSCORE                      | {nlg_results['BERTSCORE']:.4f} |")
    if 'BLEURT' in nlg_results:
        print(f"| BLEURT                         | {nlg_results['BLEURT']:.4f} |")
        print(f"| BLEURT_Composite               | {nlg_results['BLEURT_Composite']:.4f} |")
    if 'AggregateComposite' in nlg_results:
        print(f"| AggregateComposite             | {nlg_results['AggregateComposite']:.4f} |")
        print(f"| AggregateScore                 | {nlg_results['AggregateScore']:.4f} |")
    
    skipped = []
    if not include_bleurt:
        skipped.append("BLEURT (--no-bleurt)")
    if not include_bertscore:
        skipped.append("BERTScore (--no-bertscore)")
    if skipped:
        print(f"\nSkipped metrics: {', '.join(skipped)}")
