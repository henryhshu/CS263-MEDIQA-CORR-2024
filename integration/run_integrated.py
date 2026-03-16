#!/usr/bin/env python3
"""
MEDIQA-CORR 2024 — Integrated Experiment Runner

Orchestrates all three approaches on the same test set and optionally
evaluates them side-by-side:

  1. Knowledge Retrieval  — RxNorm RAG via the pipeline/ framework
  2. In-Context Learning  — baseline / fixed / dynamic via in-context-learning/
  3. Multi-Agent          — Detector → Editor → Critic via multi-agent/

Each method writes a submission file compatible with evaluation/evaluate.py.
No original module files are modified — only called through their public functions.

Usage examples:
  python integration/run_integrated.py --methods all
  python integration/run_integrated.py --methods rag icl --icl-modes fixed dynamic --model gpt-4.1
  python integration/run_integrated.py --methods multi-agent --model gpt-4.1 --no-bleurt
  python integration/run_integrated.py --methods icl --icl-modes dynamic --no-eval
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Repo root on sys.path so pipeline/ and evaluation/ are importable ─────────
# Script lives in integration/, so parent.parent is the repo root
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

INDICES_FILE = REPO_ROOT / "baseline-experiment" / "sampled_test_indices.json"
INTEGRATION_OUTPUTS = Path(__file__).parent / "outputs"


# ── API key bootstrap ─────────────────────────────────────────────────────────

def _bootstrap_openai_key() -> None:
    """Set OPENAI_API_KEY from the standard key file if not already in env."""
    if os.environ.get("OPENAI_API_KEY"):
        return
    key_path = Path.home() / "env" / "openai_secret_key.txt"
    if key_path.exists():
        key = key_path.read_text(encoding="utf-8").strip()
        if key:
            os.environ["OPENAI_API_KEY"] = key
            print(f"  (loaded OPENAI_API_KEY from {key_path})")


# ── Dataset helpers ───────────────────────────────────────────────────────────

def load_sampled_indices() -> List[int]:
    return json.loads(INDICES_FILE.read_text(encoding="utf-8"))["indices"]


def load_hf_dataset(dataset_name: str, split: str, indices: Optional[List[int]]):
    """Load a HuggingFace dataset split, optionally restricted to sampled indices."""
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split=split)
    if indices is not None:
        ds = ds.select(indices)
    return ds


# ── Dynamic module loaders ────────────────────────────────────────────────────

def _load_module_from_file(name: str, filepath: Path):
    """Import a Python file as a module regardless of its filename."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_icl_module():
    """Load in-context-learning/in-context-learning.py as a module."""
    return _load_module_from_file(
        "in_context_learning",
        REPO_ROOT / "in-context-learning" / "in-context-learning.py",
    )


def load_multiagent_module():
    """Load multi-agent/multi-agent-detect-critic-edit.py as a module.

    NOTE: This executes module-level code that reads the OpenAI API key from
    ~/env/openai_secret_key.txt (the multi-agent module's own key loading).
    Call _bootstrap_openai_key() first so it succeeds or fails gracefully.
    """
    return _load_module_from_file(
        "multi_agent_dce",
        REPO_ROOT / "multi-agent" / "multi-agent-detect-critic-edit.py",
    )


def load_evaluate_module():
    """Load evaluation/evaluate.py as a module."""
    return _load_module_from_file(
        "evaluate",
        REPO_ROOT / "evaluation" / "evaluate.py",
    )


# ── Method: Knowledge Retrieval (RxNorm RAG) ──────────────────────────────────

def run_rag(
    dataset,
    model: str,
    augmenter_type: str,
    output_path: Path,
    rate_limit_delay: float = 1.0,
) -> Path:
    """Run RxNorm RAG using the pipeline/ framework (original logic untouched)."""
    from pipeline.base import MedicalTextItem, Predictor, format_submission_line
    from pipeline.providers import OpenAIProvider
    from pipeline.augmenters import RxNormAugmenter

    provider = OpenAIProvider(model_name=model)
    augmenter = RxNormAugmenter(extractor_type=augmenter_type)
    predictor = Predictor(provider=provider, augmenter=augmenter)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(dataset)
    with output_path.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(dataset):
            item = MedicalTextItem(
                text_id=row["text_id"],
                text=row["text"],
                sentences=row["sentences"],
            )
            pred = predictor.predict(item)
            line = format_submission_line(pred)
            fh.write(line + "\n")
            drugs = pred.metadata.get("extracted_drugs", [])
            print(f"  [rag] {i+1}/{n}  {row['text_id']}  flag={pred.error_flag}  drugs={drugs}")
            if i < n - 1:
                time.sleep(rate_limit_delay)

    print(f"  → written: {output_path}")
    return output_path


# ── Method: In-Context Learning ───────────────────────────────────────────────

def run_icl(
    test_items_dicts: List[Dict[str, Any]],
    mode: str,
    model: str,
    k_shot: int,
    output_dir: Path,
    icl_train_dataset: str = "mkieffer/MEDEC",
    icl_train_split: str = "train",
    icl_train_dataset_config: str = "default",
) -> Path:
    """Run one ICL mode via the ICL module's run_mode() (original logic untouched)."""
    icl = load_icl_module()

    # Load training examples using the ICL module's own loader
    train_limit = None if mode == "dynamic" else k_shot
    train_items = icl.load_hf_dataset_split(
        icl_train_dataset,
        icl_train_dataset_config,
        icl_train_split,
        limit=train_limit,
    )

    cache_dir = REPO_ROOT / "in-context-learning" / "cache"
    output_path = icl.run_mode(
        mode=mode,
        test_items=test_items_dicts,
        train_items=train_items,
        output_dir=output_dir,
        model=model,
        k_shot=k_shot,
        embedding_model="text-embedding-3-small",
        cache_dir=cache_dir,
        reasoning_effort=None,
    )

    print(f"  → written: {output_path}")
    return output_path


# ── Method: Multi-Agent ───────────────────────────────────────────────────────

def run_multi_agent(
    dataset,
    model: str,
    output_path: Path,
    n_best: int = 3,
) -> Path:
    """Run Detector→Editor→Critic using the multi-agent module's run_one()
    (original logic untouched; dataset loading is done here to avoid the
    hardcoded path in the module's main())."""
    ma = load_multiagent_module()
    from openai import OpenAI

    client = OpenAI()
    cfg = ma.Config(
        detector_model=model,
        critic_model=model,
        editor_model=model,
        n_best=n_best,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(dataset)
    with output_path.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(dataset):
            try:
                line = ma.run_one(client, dict(row), cfg)
            except Exception as exc:
                print(f"    WARN: {row.get('text_id', '?')} failed — {exc}")
                line = ma.to_submission_line(row.get("text_id", "UNKNOWN"), 0, -1, "NA")
            fh.write(line + "\n")
            print(f"  [multi-agent] {i+1}/{n}  {line[:80]}")

    print(f"  → written: {output_path}")
    return output_path


# ── Method: Combined (RAG + ICL + Multi-Agent per item) ──────────────────────

def run_combined(
    dataset,
    model: str,
    k_shot: int,
    rag_augmenter_type: str,
    output_path: Path,
    icl_train_dataset: str = "mkieffer/MEDEC",
    icl_train_split: str = "train",
    icl_train_dataset_config: str = "default",
) -> Path:
    """
    For each item, run RAG + ICL + Multi-Agent together via CombinedPredictor.
    RAG enriches the agent instructions with drug context.
    ICL prepends k similar training examples to the Detector input.
    Multi-Agent orchestrates Detector → Critic → Editor → Critic.
    """
    from pipeline.combined import CombinedPredictor

    predictor = CombinedPredictor(
        model=model,
        k_shot=k_shot,
        rag_extractor_type=rag_augmenter_type,
    )

    # Load and embed training data for ICL retrieval
    icl = load_icl_module()
    train_items = icl.load_hf_dataset_split(
        icl_train_dataset, icl_train_dataset_config, icl_train_split
    )
    predictor.load_train_data(train_items)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(dataset)
    with output_path.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(dataset):
            try:
                line = predictor.predict(dict(row))
            except Exception as exc:
                print(f"    WARN: {row.get('text_id', '?')} failed — {exc}")
                line = f"{row.get('text_id', 'UNKNOWN')} 0 -1 NA"
            fh.write(line + "\n")
            print(f"  [combined] {i+1}/{n}  {line[:80]}")

    print(f"  → written: {output_path}")
    return output_path


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_submission(
    submission_path: Path,
    ref_dataset,
    no_bleurt: bool = False,
    no_bertscore: bool = False,
) -> Dict[str, float]:
    """Evaluate a submission file using evaluation/evaluate.py's functions."""
    ev = load_evaluate_module()

    ref_corrections, ref_flags, ref_sent_id = ev.parse_reference_dataset(ref_dataset)
    cand_corrections, cand_flags, cand_sent_id = ev.parse_run_submission_file(
        str(submission_path)
    )

    accuracy = ev.compute_accuracy(ref_flags, ref_sent_id, cand_flags, cand_sent_id)
    references, predictions, counters = ev.get_nlg_eval_data(
        ref_corrections, cand_corrections
    )
    nlg = ev.compute_nlg_metrics(
        references,
        predictions,
        counters,
        include_bleurt=not no_bleurt,
        include_bertscore=not no_bertscore,
    )
    return {**accuracy, **nlg}


def print_comparison_table(results: Dict[str, Dict[str, float]]) -> None:
    """Print a side-by-side comparison table of all evaluated methods."""
    metrics = [
        ("FlagAcc",        "Error Flags Accuracy"),
        ("SentenceAcc",    "Error Sentence Detection Accuracy"),
        ("ROUGE-1",        "ROUGE1"),
        ("ROUGE-L",        "ROUGEL"),
        ("BERTScore",      "BERTSCORE"),
        ("BLEURT",         "BLEURT"),
        ("AggScore",       "AggregateScore"),
        ("AggComposite",   "AggregateComposite"),
    ]
    col_w = 28
    hdr = f"{'Method':<{col_w}}" + "".join(f"{label:>14}" for label, _ in metrics)
    print(hdr)
    print("-" * len(hdr))
    for method, res in results.items():
        row = f"{method:<{col_w}}"
        for _label, key in metrics:
            val = res.get(key)
            row += "           N/A" if val is None else f"{val:>14.4f}"
        print(row)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="MEDIQA-CORR 2024 Integrated Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--methods", nargs="+",
        choices=["rag", "icl", "multi-agent", "combined", "all"],
        default=["all"],
        help=(
            "Method(s) to run. Use 'combined' to run RAG+ICL+Multi-Agent together "
            "per item. Use 'all' to run all three independently (default: all)"
        ),
    )
    parser.add_argument(
        "--model", default="gpt-4.1",
        help="LLM model name used by all methods (default: gpt-4.1)",
    )
    parser.add_argument(
        "--rag-augmenter", default="pubmedbert",
        choices=["pubmedbert", "regex"],
        help="RxNorm drug extractor for the RAG method (default: pubmedbert)",
    )
    parser.add_argument(
        "--icl-modes", nargs="+",
        choices=["baseline", "fixed", "dynamic"],
        default=["baseline", "fixed", "dynamic"],
        help="ICL modes to run (default: all three)",
    )
    parser.add_argument(
        "--icl-k", type=int, default=5,
        help="k for fixed/dynamic ICL (default: 5)",
    )
    parser.add_argument(
        "--dataset", default="mkieffer/MEDEC-MS",
        help="HuggingFace dataset for test inference + evaluation (default: mkieffer/MEDEC-MS)",
    )
    parser.add_argument(
        "--split", default="test",
        help="Dataset split (default: test)",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Use the full test set instead of the 50-sample subset",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=INTEGRATION_OUTPUTS,
        help=f"Base output directory (default: {INTEGRATION_OUTPUTS})",
    )
    parser.add_argument(
        "--no-eval", action="store_true",
        help="Skip evaluation after running (just produce submission files)",
    )
    parser.add_argument(
        "--no-bleurt", action="store_true",
        help="Skip BLEURT in evaluation (faster, avoids heavy model loading)",
    )
    parser.add_argument(
        "--no-bertscore", action="store_true",
        help="Skip BERTScore in evaluation (faster)",
    )
    parser.add_argument(
        "--rate-limit", type=float, default=1.0,
        help="Seconds to sleep between RAG API calls (default: 1.0)",
    )
    parser.add_argument(
        "--ma-n-best", type=int, default=3,
        help="n-best proposals for the multi-agent editor (default: 3)",
    )
    args = parser.parse_args()

    # Expand "all" into all three methods (individual), not combined
    methods = ["rag", "icl", "multi-agent"] if "all" in args.methods else list(args.methods)

    # Bootstrap API key before any method runs
    _bootstrap_openai_key()

    # Timestamped output directory for this run
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    # Sampled indices (or None for full set)
    indices = None if args.full else load_sampled_indices()

    # Load the reference test dataset once (shared by all methods + evaluation)
    print(f"\nLoading test dataset: {args.dataset} / {args.split}")
    ref_dataset = load_hf_dataset(args.dataset, args.split, indices)
    print(f"  {len(ref_dataset)} samples")

    # For ICL: convert HF rows to the dict format the ICL module expects
    icl_test_items: List[Dict[str, Any]] = []
    if "icl" in methods:
        icl = load_icl_module()
        for row in ref_dataset:
            icl_test_items.append(icl.canonicalize_item(dict(row)))

    outputs: Dict[str, Optional[Path]] = {}

    # ─ Knowledge Retrieval (RAG) ────────────────────────────────────────────
    if "rag" in methods:
        print(f"\n{'='*60}")
        print(f"Running: Knowledge Retrieval (RxNorm RAG, extractor={args.rag_augmenter})")
        print("=" * 60)
        rag_path = run_dir / f"rag_{args.rag_augmenter}_{args.model}.txt"
        try:
            outputs["rag"] = run_rag(
                dataset=ref_dataset,
                model=args.model,
                augmenter_type=args.rag_augmenter,
                output_path=rag_path,
                rate_limit_delay=args.rate_limit,
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            outputs["rag"] = None

    # ─ In-Context Learning ──────────────────────────────────────────────────
    if "icl" in methods:
        icl_out_dir = run_dir / "icl"
        for mode in args.icl_modes:
            print(f"\n{'='*60}")
            print(f"Running: In-Context Learning (mode={mode}, k={args.icl_k})")
            print("=" * 60)
            try:
                outputs[f"icl_{mode}"] = run_icl(
                    test_items_dicts=icl_test_items,
                    mode=mode,
                    model=args.model,
                    k_shot=args.icl_k,
                    output_dir=icl_out_dir,
                )
            except Exception as exc:
                print(f"  ERROR: {exc}")
                outputs[f"icl_{mode}"] = None

    # ─ Combined (RAG + ICL + Multi-Agent per item) ──────────────────────────
    if "combined" in methods:
        print(f"\n{'='*60}")
        print(f"Running: Combined (RAG + ICL k={args.icl_k} + Multi-Agent, model={args.model})")
        print("=" * 60)
        combined_path = run_dir / f"combined_rag_icl_k{args.icl_k}_ma_{args.model}.txt"
        try:
            outputs["combined"] = run_combined(
                dataset=ref_dataset,
                model=args.model,
                k_shot=args.icl_k,
                rag_augmenter_type=args.rag_augmenter,
                output_path=combined_path,
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            outputs["combined"] = None

    # ─ Multi-Agent ──────────────────────────────────────────────────────────
    if "multi-agent" in methods:
        print(f"\n{'='*60}")
        print(f"Running: Multi-Agent (Detector→Editor→Critic, model={args.model})")
        print("=" * 60)
        ma_path = run_dir / f"multi_agent_{args.model}.txt"
        try:
            outputs["multi-agent"] = run_multi_agent(
                dataset=ref_dataset,
                model=args.model,
                output_path=ma_path,
                n_best=args.ma_n_best,
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            outputs["multi-agent"] = None

    # ─ Output summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Output files written:")
    for name, path in outputs.items():
        print(f"  {name:<28} {path or 'FAILED'}")

    # ─ Evaluation ───────────────────────────────────────────────────────────
    if not args.no_eval:
        print(f"\n{'='*60}")
        print("Running Evaluation")
        print("=" * 60)

        eval_results: Dict[str, Dict[str, float]] = {}
        for name, path in outputs.items():
            if path is None or not path.exists():
                print(f"\n  Skipping {name}: output file not available")
                continue
            print(f"\n  Evaluating: {name} ({path.name})")
            try:
                metrics = evaluate_submission(
                    path,
                    ref_dataset,
                    no_bleurt=args.no_bleurt,
                    no_bertscore=args.no_bertscore,
                )
                eval_results[name] = metrics
                # Quick per-method summary
                print(f"    Error Flag Acc  : {metrics.get('Error Flags Accuracy', float('nan')):.4f}")
                print(f"    Sentence Acc    : {metrics.get('Error Sentence Detection Accuracy', float('nan')):.4f}")
                if "ROUGE1" in metrics:
                    print(f"    ROUGE-1         : {metrics['ROUGE1']:.4f}")
                if "BERTSCORE" in metrics:
                    print(f"    BERTScore       : {metrics['BERTSCORE']:.4f}")
                if "AggregateScore" in metrics:
                    print(f"    AggregateScore  : {metrics['AggregateScore']:.4f}")
            except Exception as exc:
                print(f"    ERROR: {exc}")

        if eval_results:
            print(f"\n{'='*60}")
            print("COMPARISON TABLE")
            print("=" * 60)
            print_comparison_table(eval_results)

            results_path = run_dir / "evaluation_results.json"
            results_path.write_text(
                json.dumps(eval_results, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"\nFull results saved to: {results_path}")
            print(f"Evaluate individual files with:")
            print(f"  python evaluation/evaluate.py --submission <path>")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
