#!/usr/bin/env python
"""
RxNorm RAG Integration Experiment

This script compares the performance of Gemini 2.5 Flash on the MEDIQA-CORR task
with and without RxNorm RAG augmentation.

Usage:
    python rxnorm_rag_experiment.py [--num_samples N] [--model MODEL_NAME]
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, asdict

from dotenv import load_dotenv
from datasets import load_dataset
import google.generativeai as genai

# Suppress noisy HTTP request logging from httpx/transformers
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from knowledge_retrieval.rxnorm_rag import RxNormRAGContext, DrugInfo


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for the experiment."""
    model_name: str = "gemini-2.5-flash"
    num_samples: Optional[int] = None  # None = all samples
    use_sampled_indices: bool = True
    sampled_indices_path: str = "../baseline-experiment/sampled_test_indices.json"
    output_dir: str = "outputs"
    rate_limit_delay: float = 1.0  # seconds between API calls
    extractor_type: str = "pubmedbert"  # "pubmedbert" or "regex"
    

# =============================================================================
# Prompts
# =============================================================================

BASELINE_SYSTEM_PROMPT = """
The following is a medical narrative about a patient. You are a skilled medical doctor reviewing the clinical text. The text is either correct or contains one error.
The text has a sentence per line. Each line starts with the sentence ID, followed by a pipe character then the sentence to check. Check every sentence of the text.
If the text is correct return the following output: CORRECT. If the text has a medical error, return the sentence id of the sentence containing the error,
followed by a space, and a corrected version of the sentence.
"""

RAG_SYSTEM_PROMPT_TEMPLATE = """
The following is a medical narrative about a patient. You are a skilled medical doctor reviewing the clinical text. The text is either correct or contains one error.
The text has a sentence per line. Each line starts with the sentence ID, followed by a pipe character then the sentence to check. Check every sentence of the text.
If the text is correct return the following output: CORRECT. If the text has a medical error, return the sentence id of the sentence containing the error,
followed by a space, and a corrected version of the sentence.

{drug_context}

Use the above drug reference information (if provided) to help identify and correct any medical errors related to medications, dosages, drug interactions, or prescribing guidelines.
"""


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PredictionResult:
    """Result from a single prediction."""
    text_id: str
    error_flag: int
    sentence_id: int
    corrected_sentence: Optional[str]
    extracted_drugs: List[str]
    raw_output: str
    
    
@dataclass
class EvaluationMetrics:
    """Evaluation metrics for a set of predictions."""
    total_samples: int
    error_flag_correct: int
    error_flag_accuracy: float
    sentence_id_correct: int
    sentence_id_accuracy: float


# =============================================================================
# Helper Functions
# =============================================================================

def load_indices(path: str) -> List[int]:
    """Load sampled indices from JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))["indices"]


def escape_for_double_quotes(s: str) -> str:
    """Escape backslashes and double quotes."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def parse_model_output(output_text: str) -> Tuple[int, int, Optional[str]]:
    """
    Parse the model output into submission fields.
    
    Returns:
        Tuple of (error_flag, sentence_id, corrected_sentence)
    """
    t = (output_text or "").strip()
    
    # Check for CORRECT response
    if t.upper().rstrip(".") == "CORRECT":
        return 0, -1, None
        
    # Try to parse "sentence_id corrected_sentence" format
    parts = t.split(None, 1)
    if not parts:
        return 0, -1, None
        
    try:
        sid = int(parts[0])
    except ValueError:
        return 0, -1, None
        
    corrected = parts[1].strip() if len(parts) > 1 else ""
    if not corrected:
        corrected = "NA"
        
    return 1, sid, corrected


def format_submission_line(text_id: str, flag: int, sid: int, corrected: Optional[str]) -> str:
    """Format a submission line."""
    if flag == 0:
        return f"{text_id} 0 -1 NA"
    else:
        corrected_escaped = escape_for_double_quotes(corrected or "NA")
        return f'{text_id} 1 {sid} "{corrected_escaped}"'


# =============================================================================
# Experiment Runner
# =============================================================================

class ExperimentRunner:
    """Runs the RAG vs baseline comparison experiment."""
    
    def __init__(self, config: ExperimentConfig):
        """Initialize the experiment runner."""
        self.config = config
        self.rag = RxNormRAGContext(extractor_type=config.extractor_type)
        
        # Load environment variables
        env_path = Path(__file__).parent.parent / ".env"
        load_dotenv(dotenv_path=env_path)
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file")
            
        genai.configure(api_key=api_key)
        
        # Create models
        self.model_baseline = genai.GenerativeModel(
            model_name=config.model_name,
            system_instruction=BASELINE_SYSTEM_PROMPT
        )
        
        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_dataset(self) -> list:
        """Load and prepare the dataset."""
        dataset = load_dataset("mkieffer/MEDEC-MS", split="test")
        
        if self.config.use_sampled_indices:
            indices_path = Path(__file__).parent / self.config.sampled_indices_path
            if indices_path.exists():
                indices = load_indices(str(indices_path))
                dataset = dataset.select(indices)
                print(f"Loaded {len(dataset)} sampled test examples")
            else:
                print(f"Warning: Sampled indices file not found at {indices_path}")
                print("Using full test set")
                
        if self.config.num_samples:
            dataset = dataset.select(range(min(self.config.num_samples, len(dataset))))
            print(f"Limited to {len(dataset)} samples")
            
        return list(dataset)
    
    def run_baseline(self, item: dict) -> PredictionResult:
        """Run baseline prediction (no RAG)."""
        text_id = item["text_id"]
        sentences = item["sentences"]
        
        try:
            response = self.model_baseline.generate_content(sentences)
            output_text = response.text
        except Exception as e:
            print(f"  Error: {e}")
            output_text = "CORRECT"
            
        flag, sid, corrected = parse_model_output(output_text)
        
        return PredictionResult(
            text_id=text_id,
            error_flag=flag,
            sentence_id=sid,
            corrected_sentence=corrected,
            extracted_drugs=[],
            raw_output=output_text
        )
    
    def run_with_rag(self, item: dict) -> PredictionResult:
        """Run prediction with RAG augmentation."""
        text_id = item["text_id"]
        text = item["text"]
        sentences = item["sentences"]
        
        # Extract drugs and build augmented prompt
        extracted_drugs = self.rag.extract_drugs_from_text(text, validate=True)
        
        if extracted_drugs:
            drug_context = self.rag.get_drug_context(extracted_drugs)
            system_prompt = RAG_SYSTEM_PROMPT_TEMPLATE.format(drug_context=drug_context)
        else:
            system_prompt = BASELINE_SYSTEM_PROMPT
            
        # Create model with augmented system prompt
        model_rag = genai.GenerativeModel(
            model_name=self.config.model_name,
            system_instruction=system_prompt
        )
        
        try:
            response = model_rag.generate_content(sentences)
            output_text = response.text
        except Exception as e:
            print(f"  Error: {e}")
            output_text = "CORRECT"
            
        flag, sid, corrected = parse_model_output(output_text)
        
        return PredictionResult(
            text_id=text_id,
            error_flag=flag,
            sentence_id=sid,
            corrected_sentence=corrected,
            extracted_drugs=extracted_drugs,
            raw_output=output_text
        )
    
    def evaluate(self, predictions: List[PredictionResult], dataset: list) -> EvaluationMetrics:
        """Evaluate predictions against ground truth."""
        pred_dict = {p.text_id: p for p in predictions}
        
        flag_correct = 0
        sid_correct = 0
        total = 0
        
        for item in dataset:
            text_id = item["text_id"]
            if text_id not in pred_dict:
                continue
                
            pred = pred_dict[text_id]
            gt_flag = 1 if item["error_flag"] else 0
            gt_sid = item["error_sentence_id"] if item["error_flag"] else -1
            
            if pred.error_flag == gt_flag:
                flag_correct += 1
                
            if pred.sentence_id == gt_sid:
                sid_correct += 1
                
            total += 1
            
        return EvaluationMetrics(
            total_samples=total,
            error_flag_correct=flag_correct,
            error_flag_accuracy=flag_correct / total if total > 0 else 0,
            sentence_id_correct=sid_correct,
            sentence_id_accuracy=sid_correct / total if total > 0 else 0
        )
    
    def save_results(
        self, 
        baseline_preds: List[PredictionResult],
        rag_preds: List[PredictionResult],
        baseline_metrics: EvaluationMetrics,
        rag_metrics: EvaluationMetrics,
        timestamp: str
    ):
        """Save experiment results to files."""
        # Save baseline predictions
        baseline_file = self.output_dir / f"baseline_{timestamp}.txt"
        with open(baseline_file, "w") as f:
            for pred in baseline_preds:
                line = format_submission_line(
                    pred.text_id, pred.error_flag, 
                    pred.sentence_id, pred.corrected_sentence
                )
                f.write(line + "\n")
                
        # Save RAG predictions
        rag_file = self.output_dir / f"rag_{timestamp}.txt"
        with open(rag_file, "w") as f:
            for pred in rag_preds:
                line = format_submission_line(
                    pred.text_id, pred.error_flag,
                    pred.sentence_id, pred.corrected_sentence
                )
                f.write(line + "\n")
                
        # Save detailed comparison
        comparison_file = self.output_dir / f"comparison_{timestamp}.json"
        comparison_data = {
            "config": asdict(self.config),
            "timestamp": timestamp,
            "baseline_metrics": asdict(baseline_metrics),
            "rag_metrics": asdict(rag_metrics),
            "improvement": {
                "error_flag_accuracy": rag_metrics.error_flag_accuracy - baseline_metrics.error_flag_accuracy,
                "sentence_id_accuracy": rag_metrics.sentence_id_accuracy - baseline_metrics.sentence_id_accuracy
            },
            "predictions": []
        }
        
        # Create prediction-level comparison
        baseline_dict = {p.text_id: p for p in baseline_preds}
        rag_dict = {p.text_id: p for p in rag_preds}
        
        for text_id in baseline_dict:
            bp = baseline_dict[text_id]
            rp = rag_dict.get(text_id)
            
            comparison_data["predictions"].append({
                "text_id": text_id,
                "baseline": {
                    "error_flag": bp.error_flag,
                    "sentence_id": bp.sentence_id,
                    "corrected": bp.corrected_sentence
                },
                "rag": {
                    "error_flag": rp.error_flag if rp else None,
                    "sentence_id": rp.sentence_id if rp else None,
                    "corrected": rp.corrected_sentence if rp else None,
                    "extracted_drugs": rp.extracted_drugs if rp else []
                },
                "different": (
                    bp.error_flag != rp.error_flag or 
                    bp.sentence_id != rp.sentence_id
                ) if rp else True
            })
            
        with open(comparison_file, "w") as f:
            json.dump(comparison_data, f, indent=2)
            
        print(f"\nResults saved to:")
        print(f"  Baseline: {baseline_file}")
        print(f"  RAG: {rag_file}")
        print(f"  Comparison: {comparison_file}")
        
    def run(self):
        """Run the full experiment."""
        print("=" * 60)
        print("RxNorm RAG Integration Experiment")
        print("=" * 60)
        print(f"Model: {self.config.model_name}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        # Load dataset
        print("\nLoading dataset...")
        dataset = self.load_dataset()
        
        # Run baseline
        print("\n" + "-" * 40)
        print("Phase 1: Running BASELINE (no RAG)")
        print("-" * 40)
        
        baseline_predictions = []
        for i, item in enumerate(dataset):
            text_id = item["text_id"]
            print(f"[{i+1}/{len(dataset)}] {text_id} (baseline)...", end=" ")
            
            pred = self.run_baseline(item)
            baseline_predictions.append(pred)
            
            print(f"flag={pred.error_flag}, sid={pred.sentence_id}")
            time.sleep(self.config.rate_limit_delay)
            
        # Evaluate baseline
        baseline_metrics = self.evaluate(baseline_predictions, dataset)
        print(f"\nBaseline Results:")
        print(f"  Error Flag Accuracy: {baseline_metrics.error_flag_accuracy:.4f}")
        print(f"  Sentence ID Accuracy: {baseline_metrics.sentence_id_accuracy:.4f}")
        
        # Run with RAG
        print("\n" + "-" * 40)
        print("Phase 2: Running WITH RAG")
        print("-" * 40)
        
        rag_predictions = []
        total_drugs_extracted = 0
        
        for i, item in enumerate(dataset):
            text_id = item["text_id"]
            print(f"[{i+1}/{len(dataset)}] {text_id} (RAG)...", end=" ")
            
            pred = self.run_with_rag(item)
            rag_predictions.append(pred)
            
            drugs_str = f", drugs={pred.extracted_drugs}" if pred.extracted_drugs else ""
            print(f"flag={pred.error_flag}, sid={pred.sentence_id}{drugs_str}")
            
            total_drugs_extracted += len(pred.extracted_drugs)
            time.sleep(self.config.rate_limit_delay)
            
        # Evaluate RAG
        rag_metrics = self.evaluate(rag_predictions, dataset)
        print(f"\nRAG Results:")
        print(f"  Error Flag Accuracy: {rag_metrics.error_flag_accuracy:.4f}")
        print(f"  Sentence ID Accuracy: {rag_metrics.sentence_id_accuracy:.4f}")
        print(f"  Total drugs extracted: {total_drugs_extracted}")
        print(f"  Avg drugs per sample: {total_drugs_extracted / len(dataset):.2f}")
        
        # Comparison
        print("\n" + "=" * 60)
        print("COMPARISON: BASELINE vs RAG")
        print("=" * 60)
        
        flag_diff = rag_metrics.error_flag_accuracy - baseline_metrics.error_flag_accuracy
        sid_diff = rag_metrics.sentence_id_accuracy - baseline_metrics.sentence_id_accuracy
        
        print(f"\n{'Metric':<35} {'Baseline':>10} {'RAG':>10} {'Δ':>10}")
        print("-" * 65)
        print(f"{'Error Flag Accuracy':<35} {baseline_metrics.error_flag_accuracy:>10.4f} {rag_metrics.error_flag_accuracy:>10.4f} {flag_diff:>+10.4f}")
        print(f"{'Sentence ID Accuracy':<35} {baseline_metrics.sentence_id_accuracy:>10.4f} {rag_metrics.sentence_id_accuracy:>10.4f} {sid_diff:>+10.4f}")
        
        # Count differences
        baseline_dict = {p.text_id: p for p in baseline_predictions}
        rag_dict = {p.text_id: p for p in rag_predictions}
        
        different_predictions = 0
        improved = 0
        degraded = 0
        
        for item in dataset:
            text_id = item["text_id"]
            bp = baseline_dict.get(text_id)
            rp = rag_dict.get(text_id)
            
            if not bp or not rp:
                continue
                
            gt_flag = 1 if item["error_flag"] else 0
            gt_sid = item["error_sentence_id"] if item["error_flag"] else -1
            
            if bp.error_flag != rp.error_flag or bp.sentence_id != rp.sentence_id:
                different_predictions += 1
                
                # Check if RAG improved or degraded
                bp_correct = (bp.error_flag == gt_flag and bp.sentence_id == gt_sid)
                rp_correct = (rp.error_flag == gt_flag and rp.sentence_id == gt_sid)
                
                if rp_correct and not bp_correct:
                    improved += 1
                elif bp_correct and not rp_correct:
                    degraded += 1
                    
        print(f"\n{'Different Predictions':<35} {different_predictions:>10}")
        print(f"{'Improved by RAG':<35} {improved:>10}")
        print(f"{'Degraded by RAG':<35} {degraded:>10}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_results(
            baseline_predictions, rag_predictions,
            baseline_metrics, rag_metrics, timestamp
        )
        
        return baseline_metrics, rag_metrics


# =============================================================================
# Main
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RxNorm RAG Experiment")
    parser.add_argument(
        "--num_samples", "-n", type=int, default=None,
        help="Number of samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--model", "-m", type=str, default="gemini-2.5-flash",
        help="Model name (default: gemini-2.5-flash)"
    )
    parser.add_argument(
        "--rate_limit", "-r", type=float, default=1.0,
        help="Delay between API calls in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--extractor", "-e", type=str, default="pubmedbert",
        choices=["pubmedbert", "regex"],
        help="Drug name extractor: 'pubmedbert' (NER model) or 'regex' (legacy). Default: pubmedbert"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run on the full test set instead of the sampled subset"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    config = ExperimentConfig(
        model_name=args.model,
        num_samples=args.num_samples,
        rate_limit_delay=args.rate_limit,
        extractor_type=args.extractor,
        use_sampled_indices=not args.full
    )
    
    runner = ExperimentRunner(config)
    runner.run()
