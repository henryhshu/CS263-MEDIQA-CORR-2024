#!/usr/bin/env python
# coding: utf-8
"""
Simplified evaluation script for Gemini results
Computes: Accuracy, ROUGE, and BERTSCORE (skips BLEURT due to compatibility issues)
"""

import re
import pandas as pd
from rouge import Rouge
import bert_score.score as bertscore
import numpy as np
import math
import string
import json
from pathlib import Path
from datasets import load_dataset

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


def compute_nlg_metrics(references, predictions, counters):
    """Compute ROUGE and BERTSCORE."""
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
    print("Computing BERTScore (this may take a few minutes)...")
    _, _, bertScore_F1 = bertscore(
        predictions, references, 
        model_type='roberta-large', 
        lang='en', 
        device='cpu',
        verbose=True, 
        rescale_with_baseline=True
    )
    
    bertscores = np.array([clip(num) for num in bertScore_F1.numpy()])
    results['BERTSCORE'] = np.mean(bertscores)
    
    # Composite scores
    rouge1_composite = (np.sum(rouge1f_scores) + counters["system_provided_correct_na"]) / counters["total_texts"]
    bertscore_composite = (np.sum(bertscores) + counters["system_provided_correct_na"]) / counters["total_texts"]
    
    # Aggregate (ROUGE1 + BERTSCORE) / 2
    aggregate_per_sample = (np.array(rouge1f_scores) + bertscores) / 2
    aggregate_composite = (np.sum(aggregate_per_sample) + counters["system_provided_correct_na"]) / counters["total_texts"]
    
    results['ROUGE1_Composite'] = rouge1_composite
    results['BERTSCORE_Composite'] = bertscore_composite
    results['AggregateComposite'] = aggregate_composite
    results['AggregateScore'] = np.mean(aggregate_per_sample)
    
    return results


def load_indices(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))["indices"]


if __name__ == "__main__":
    # Configuration
    submission_file = "../baseline-experiment/outputs/medec-ms_gemini-2.5-flash_results_20260206_192929.txt"
    indices_file = "../baseline-experiment/sampled_test_indices.json"
    split = "test"
    
    print("=" * 60)
    print("MEDIQA-CORR Evaluation for Gemini 2.5 Flash")
    print("=" * 60)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset("mkieffer/MEDEC", split=split)
    loaded_indices = load_indices(indices_file)
    dataset = dataset.select(loaded_indices)
    print(f"Loaded {len(dataset)} samples")
    
    # Parse reference and predictions
    reference_corrections, reference_flags, reference_sent_id = parse_reference_dataset(dataset)
    candidate_corrections, candidate_flags, candidate_sent_id = parse_run_submission_file(submission_file)
    
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
    
    nlg_results = compute_nlg_metrics(references, predictions, counters)
    print("\nResults:")
    for k, v in nlg_results.items():
        print(f"  {k}: {v:.4f}")
    
    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY - Gemini 2.5 Flash Results")
    print("=" * 60)
    print(f"| Metric                          | Value  |")
    print(f"|--------------------------------|--------|")
    print(f"| Error Flag Accuracy            | {accuracy_results['Error Flags Accuracy']:.4f} |")
    print(f"| Error Sentence Detection Acc   | {accuracy_results['Error Sentence Detection Accuracy']:.4f} |")
    print(f"| ROUGE1                         | {nlg_results['ROUGE1']:.4f} |")
    print(f"| BERTSCORE                      | {nlg_results['BERTSCORE']:.4f} |")
    print(f"| AggregateComposite             | {nlg_results['AggregateComposite']:.4f} |")
    print(f"| AggregateScore                 | {nlg_results['AggregateScore']:.4f} |")
    print("\nNote: BLEURT metric skipped due to package compatibility issues.")
